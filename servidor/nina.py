# Nina, a agente de IA do site murinoi.com — roda na própria VPS, sem n8n e sem mensalidade.
# Fala direto com o Gemini (Google). A chave fica só na VPS: cadastre em https://murinoi.com/nina/chave?c=CÓDIGO
# (o código aparece no log do instalador). Este arquivo é baixado do GitHub e se atualiza sozinho.
#
# Rotas (atrás do Caddy, em /nina/...):
#   POST /nina/chat    conversa do chat do site (mesmo formato que o site já usa)
#   POST /nina/ouvir   transcrição do áudio do microfone (Gemini, palavra por palavra)
#   POST /nina/voz     voz da Nina pelo Gemini TTS (reserva da voz principal /voz)
#   POST /nina/bot     chatbot do WhatsApp (chamado pela Evolution API, só na rede interna)
#   GET  /nina/saude   o site pergunta se a IA está pronta
#   GET  /nina/chave?c=…  cadastrar/trocar a chave do Gemini
#   GET  /nina/leads?c=…  ver os clientes atendidos pela Nina
import asyncio, ast, base64, datetime, hashlib, html, json, math, operator, os, re, subprocess, tempfile, time
import aiohttp
from aiohttp import web

DADOS = os.environ.get('DADOS', '/dados'); os.makedirs(DADOS, exist_ok=True)
COD = os.environ.get('CODIGO', '')
API = os.environ.get('GEMINI_URL', 'https://generativelanguage.googleapis.com/v1beta/models/')
EVO_URL = os.environ.get('EVO_URL', 'http://murinoi-evo:8080')
NUM_FONTE = os.environ.get('NUM_FONTE', 'https://raw.githubusercontent.com/wyterlin1-svg/Web/hostinger/index.html')
ORIGENS = re.compile(r'^https://(www\.)?murinoi\.com(/|$)|^http://(localhost|127\.0\.0\.1)(:\d+)?(/|$)')
MODELOS_CHAT = ['gemini-3.1-flash-lite-preview', 'gemini-2.5-flash-lite', 'gemini-2.5-flash']
MODELOS_OUVIR = ['gemini-2.5-flash', 'gemini-3.1-flash-lite-preview', 'gemini-2.5-flash-lite']
MODELO_VOZ = 'gemini-2.5-flash-preview-tts'
BRT = datetime.timezone(datetime.timedelta(hours=-3))   # horário de Brasília (sem horário de verão)

def log(*a): print(time.strftime('%d/%m %H:%M'), *a, flush=True)

# ---------------- chave do Gemini ----------------
ARQ_CHAVE = DADOS + '/gemini_key'
def chave():
    try: return open(ARQ_CHAVE).read().strip()
    except Exception: return os.environ.get('GEMINI_KEY', '').strip()

def config_modelo(modelo, base):
    c = dict(base)
    if modelo.startswith('gemini-2.5-flash') and 'lite' not in modelo and 'tts' not in modelo:
        c['thinkingConfig'] = {'thinkingBudget': 0}   # resposta rápida, sem "pensar" à toa
    return c

async def gemini(modelo, corpo, espera=30, k=None):
    k = k or chave()
    if not k: raise RuntimeError('sem chave do Gemini')
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=espera)) as s:
        async with s.post(API + modelo + ':generateContent', json=corpo, headers={'x-goog-api-key': k}) as r:
            d = await r.json(content_type=None)
            if r.status != 200: raise RuntimeError('%s %s' % (r.status, str(d)[:300]))
            return d

def texto_de(d):
    partes = (((d.get('candidates') or [{}])[0].get('content') or {}).get('parts') or [])
    return ''.join(p.get('text', '') for p in partes if not p.get('thought')).strip()

# ---------------- limites por IP ----------------
vistos = {}
def limite(req, grupo, n, janela=60):
    ip = req.headers.get('X-Forwarded-For', req.remote or '').split(',')[0].strip()
    agora = time.time(); ch = grupo + ip
    h = [x for x in vistos.get(ch, []) if agora - x < janela]
    if len(h) >= n: raise web.HTTPTooManyRequests()
    vistos[ch] = h + [agora]
    if len(vistos) > 20000: vistos.clear()
    return ip

def do_site(req):
    o = req.headers.get('Origin') or req.headers.get('Referer') or ''
    if not ORIGENS.match(o): raise web.HTTPForbidden()

# ---------------- calculadora (contas exatas, nunca de cabeça) ----------------
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}
def calcular(expr):
    expr = str(expr).replace('×', '*').replace('÷', '/').replace('^', '**')
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 12: raise ValueError('expoente grande')
            return OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.operand))
        raise ValueError('conta inválida')
    v = ev(ast.parse(expr, mode='eval'))
    if not math.isfinite(v): raise ValueError('resultado inválido')
    return v

def br_num(v, casas=2):
    s = ('{:,.%df}' % casas).format(v)
    return s.replace(',', '#').replace('.', ',').replace('#', '.')

def formatar(v, formato):
    if formato == 'reais': return 'R$ ' + br_num(v, 2)
    if formato == 'horas': return br_num(v, 1).rstrip('0').rstrip(',') + (' hora' if abs(v - 1) < 1e-9 else ' horas')
    if formato == 'percentual': return br_num(v, 1).rstrip('0').rstrip(',') + '%'
    if formato == 'meses': return br_num(v, 1).rstrip('0').rstrip(',') + (' mês' if abs(v - 1) < 1e-9 else ' meses')
    return br_num(v, 2).rstrip('0').rstrip(',')

def aplicar_contas(o):
    valores = {}
    for c in o.get('calculos') or []:
        try: valores[str(c.get('id', '')).strip('{} ')] = formatar(calcular(c.get('expressao', '')), c.get('formato', 'numero'))
        except Exception: pass
    def troca(t):
        return re.sub(r'\{\{?\s*(c\d+)\s*\}?\}', lambda m: valores.get(m.group(1), m.group(0)), t or '')
    for campo in ('resposta', 'solucao', 'resumo'):
        if o.get(campo): o[campo] = troca(o[campo])
    return not re.search(r'\{\{?\s*c\d+\s*\}?\}', o.get('resposta', ''))   # True = todas as contas resolvidas

# ---------------- instruções da Nina (chat do site) ----------------
SISTEMA_SITE = """Você é a Nina, consultora de soluções com inteligência artificial da Murinoi, empresa brasileira que cria soluções sob medida para negócios e pessoas: aplicativos (inclusive na Google Play), sistemas, sites, automações, integrações e soluções com IA. Promessa: menos custo e mais tempo para o negócio do cliente.

Você conversa no chat do site murinoi.com. Seu trabalho é entender a necessidade por completo, medir o problema com números e montar a melhor proposta de solução, para o especialista já chegar com a resposta certa.

## O que descobrir (uma pergunta por vez, sem repetir o que já foi dito)
1. Tipo de negócio e porte (quantas pessoas, unidades ou atendimentos).
2. O problema: o que toma tempo ou dinheiro e como é feito hoje (ferramentas, planilhas, papel, WhatsApp).
3. Os números do problema: quantas vezes por dia/mês, quanto tempo leva, quantas pessoas, ticket médio ou salário médio. Peça no máximo 2 números, de forma simples.
4. O resultado esperado e o prazo ou urgência.
5. Nome do cliente.
6. Melhor contato (WhatsApp com DDD ou e-mail).

## Método da proposta
- Diagnóstico: em uma frase, o que trava o negócio.
- Conta: quando tiver números, calcule o custo do problema e o ganho possível. NUNCA escreva o resultado de uma conta de cabeça: coloque a conta no campo "calculos" (com id c1, c2…, a expressão com números e + - * / e parênteses, ponto como separador decimal, e o formato: reais, horas, percentual, meses ou numero) e, no texto, escreva o marcador {c1}, {c2}… no lugar do resultado; o sistema faz a conta exata e troca o marcador pelo valor já formatado (ex.: R$ 1.234,56). Deixe claro que é uma estimativa e quais hipóteses usou.
- Solução: diga qual solução da Murinoi resolve (app, sistema, automação, integração, IA ou site) e o que ela faz na prática.
- Próximo passo: a equipe envia a proposta com solução, prazo e investimento.

## Fórmulas (monte a expressão com elas)
- Custo da hora de um funcionário CLT ≈ salário mensal × 1,7 (encargos e benefícios, aproximado) ÷ 176 horas úteis por mês.
- Horas gastas por mês = minutos por tarefa × vezes por dia × 22 dias úteis ÷ 60.
- Economia mensal = horas economizadas × custo da hora (+ erros evitados × custo de cada erro). Economia anual = mensal × 12.
- Receita recuperada = vendas ou atendimentos perdidos × ticket médio × percentual recuperado (use hipótese conservadora e diga isso).
- Ticket médio = faturamento ÷ número de vendas. Margem = lucro ÷ receita × 100.
- ROI = (ganho no período − investimento) ÷ investimento × 100. Payback em meses = investimento ÷ ganho mensal (só calcule se o cliente informar um valor de investimento; nunca invente preço da Murinoi).
- Ponto de equilíbrio = custos fixos ÷ (1 − custos variáveis ÷ receita).

## Conhecimento por setor
Financeiro (empresas, financeiras, fintechs, cobrança): fluxo de caixa realizado e projetado, DRE gerencial, contas a pagar e a receber, conciliação bancária automática por API/extrato (OFX) e Open Finance, Pix (QR dinâmico, Pix Cobrança e Pix Automático), boletos registrados, régua de cobrança automática por WhatsApp e e-mail, análise de crédito e score, painéis (dashboards) de indicadores. Indicadores: inadimplência (% = vencido ÷ faturado × 100), prazo médio de recebimento (contas a receber ÷ faturamento × 30), capital de giro, margem de contribuição. Para instituições reguladas, lembre de segurança, trilha de auditoria, prevenção à lavagem de dinheiro (PLD/KYC) e normas do Banco Central.

Contábil (escritórios e departamentos fiscais): captura e leitura automática de notas (NF-e, NFS-e, NFC-e, XML), classificação e lançamentos contábeis, conciliação, folha e eSocial, obrigações acessórias (SPED/EFD, DCTFWeb), portal do cliente para envio de documentos, cobrança de honorários, alertas de prazos e IA para triagem de documentos. Reforma Tributária (IBS e CBS) com transição gradual a partir de 2026: sistemas precisam se adaptar às novas regras. Indicadores: horas de digitação por cliente, clientes por contador, erros e multas evitadas.

Restaurantes, bares, lanchonetes e delivery: PDV/frente de caixa, comanda digital, KDS (tela na cozinha), cardápio digital por QR Code, pedidos online e delivery próprio integrado ao WhatsApp, integração com marketplaces, ficha técnica e CMV, controle de estoque e desperdício, emissão de NFC-e, reservas, fidelidade e cashback. Indicadores: CMV % (custo dos insumos ÷ receita × 100), ticket médio, giro de mesas, tempo de preparo, economia de comissão (pedidos no canal próprio × ticket médio × comissão do marketplace informada pelo cliente).

Saúde (hospitais, clínicas, consultórios, laboratórios): agendamento online, confirmação e lembretes automáticos para reduzir faltas, prontuário eletrônico seguindo as exigências do CFM, faturamento de convênios no padrão TISS/TUSS da ANS e redução de glosas, gestão de leitos e fila, estoque de farmácia e materiais, integração com laboratórios, telemedicina e painéis de gestão. Dados de saúde são sensíveis na LGPD: sempre cite segurança e controle de acesso. Indicadores: taxa de faltas (faltas ÷ agendamentos × 100), receita perdida com faltas (faltas por mês × valor da consulta), taxa de glosa (glosado ÷ faturado × 100), taxa de ocupação (pacientes-dia ÷ leitos-dia × 100), tempo médio de permanência.

Outros setores (comércio, serviços, indústria, educação, logística, imóveis): aplique o mesmo método, identificando a tarefa repetitiva, o volume e o custo.

## Fechamento
Quando tiver os itens 1, 2, 5 e 6: marque pronto = true e preencha nome, contato, negocio, necessidade, resultado, prazo, solucao e resumo. Na solucao, descreva a solução recomendada e o ganho estimado. O resumo (6 a 10 linhas, para a equipe) deve ter: contexto e porte, dor atual, números informados, contas feitas com as hipóteses, solução recomendada com módulos principais, ganho estimado e urgência. Na resposta, agradeça pelo nome, confirme o que entendeu em uma frase, diga que a equipe da Murinoi vai entrar em contato com a solução, o prazo e o investimento e termine sempre perguntando se pode ajudar com mais alguma coisa (por exemplo: "Posso te ajudar com mais alguma coisa?"). Antes disso, pronto = false.

Depois que o atendimento estiver pronto, continue com pronto = true e mantenha os campos preenchidos (atualize-os se o cliente trouxer informação nova). Se o cliente pedir mais alguma coisa, ajude normalmente. Se ele disser que não precisa de mais nada ou agradecer, despeça-se com gentileza em uma frase, sem fazer nova pergunta.

Pedido de atendente humano: se o cliente pedir para falar com uma pessoa, atendente, humano, vendedor ou especialista, marque etapa_humano = "pediu" (o sistema troca a sua resposta por uma pergunta padrão pedindo que ele explique o pedido). Na mensagem seguinte do cliente, que é a explicação, marque etapa_humano = "explicou" e preencha necessidade e resumo com o que ele explicou, além de nome, contato e negócio se já souber. Em todos os outros casos use etapa_humano = "nenhuma".

## Estilo
Português do Brasil, cordial, leve e profissional, com segurança de quem entende do assunto. No máximo 3 frases curtas por resposta (até 4 quando mostrar uma conta). Sem markdown, títulos ou listas. No máximo 1 emoji. Use os termos do setor do cliente com naturalidade, sem jargão desnecessário.

## Regras
- Nunca invente preços da Murinoi, prazos fechados, clientes, casos ou estatísticas. Se perguntarem valor, diga que depende do escopo e que a equipe envia a proposta sem compromisso.
- Números só vêm do cliente ou de hipóteses que você deixa explícitas como estimativa.
- Nunca peça senhas, documentos, dados de cartão ou dados de pacientes.
- Assuntos fora da Murinoi recebem uma resposta gentil de uma frase e você volta ao negócio do cliente.
- O e-mail da Murinoi é contato@murinoi.com."""

S = lambda d='': {'type': 'STRING', 'description': d} if d else {'type': 'STRING'}
ESQUEMA = {'type': 'OBJECT', 'properties': {
    'resposta': S('Mensagem para o cliente, curta, em português do Brasil (com {c1}… no lugar dos resultados das contas)'),
    'pronto': {'type': 'BOOLEAN', 'description': 'true somente quando já entendeu a necessidade e tem nome e um contato'},
    'etapa_humano': {'type': 'STRING', 'enum': ['nenhuma', 'pediu', 'explicou']},
    'calculos': {'type': 'ARRAY', 'items': {'type': 'OBJECT', 'properties': {
        'id': S('c1, c2…'), 'expressao': S('Ex.: (3000*1.7/176)*(10*20*22/60)'),
        'formato': {'type': 'STRING', 'enum': ['reais', 'horas', 'percentual', 'meses', 'numero']}}, 'required': ['id', 'expressao', 'formato']}},
    'nome': S(), 'contato': S('WhatsApp com DDD ou e-mail informado pelo cliente'), 'negocio': S(),
    'necessidade': S('O problema do cliente e como ele faz hoje (ou a explicação do pedido de atendente)'),
    'resultado': S('O resultado que o cliente espera alcançar'), 'prazo': S('Prazo ou urgência (vazio se não informou)'),
    'solucao': S('Solução sugerida pela Murinoi, com o ganho esperado'), 'resumo': S('Resumo executivo completo para a equipe')},
    'required': ['resposta', 'pronto', 'etapa_humano']}

def conversa(historico, atual, cabecalho):
    itens = []
    for m in (historico if isinstance(historico, list) else [])[-40:]:
        if not isinstance(m, dict): continue
        t = str(m.get('t') or '').strip()[:1500]
        if t: itens.append(['user' if m.get('q') == 'eu' else 'model', t])
    if itens and itens[-1][0] == 'user' and itens[-1][1] == atual.strip(): itens.pop()   # o site já manda a mensagem atual no histórico
    itens.append(['user', cabecalho + atual])
    if itens[0][0] != 'user': itens.insert(0, ['user', 'Olá'])
    juntos = []
    for papel, t in itens:   # o Gemini quer papéis alternados
        if juntos and juntos[-1]['role'] == papel: juntos[-1]['parts'][0]['text'] += '\n' + t
        else: juntos.append({'role': papel, 'parts': [{'text': t}]})
    return juntos

def ler_json(t):
    t = re.sub(r'^```(?:json)?|```$', '', t.strip()).strip()
    try: return json.loads(t)
    except Exception:
        m = re.search(r'\{.*\}', t, re.S)
        return json.loads(m.group(0)) if m else None

async def pensar_site(historico, msg):
    agora = datetime.datetime.now(BRT).strftime('%d/%m/%Y %H:%M')
    corpo_base = {'systemInstruction': {'parts': [{'text': SISTEMA_SITE}]},
                  'contents': conversa(historico, msg, 'Data e hora: %s\n\nMensagem do visitante:\n' % agora)}
    erro = None
    for modelo in MODELOS_CHAT:
        for tentativa in range(2):
            try:
                corpo = dict(corpo_base, generationConfig=config_modelo(modelo, {
                    'temperature': 0.3, 'maxOutputTokens': 4096, 'responseMimeType': 'application/json', 'responseSchema': ESQUEMA}))
                o = ler_json(texto_de(await gemini(modelo, corpo, 40)))
                if isinstance(o, dict) and str(o.get('resposta', '')).strip():
                    if aplicar_contas(o) or tentativa: return o
            except Exception as e:
                erro = e; log('chat: %s falhou: %s' % (modelo, str(e)[:200]))
                if 'sem chave' in str(e): raise
                break
    raise RuntimeError('IA indisponível: %s' % erro)

# ---------------- leads (substitui a tabela do n8n) ----------------
ARQ_LEADS = DADOS + '/leads.json'; trava_leads = asyncio.Lock()
def ler_leads():
    try: return json.load(open(ARQ_LEADS))
    except Exception: return {}
def gravar_leads(d):
    with open(ARQ_LEADS + '.tmp', 'w') as f: json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(ARQ_LEADS + '.tmp', ARQ_LEADS)

def protocolo(sessao):
    return 'MUR-' + datetime.datetime.now(BRT).strftime('%y%m%d') + '-' + re.sub(r'[^a-z0-9]', '', sessao, flags=re.I)[-4:].upper()

async def salvar_lead(sessao, o, historico, pagina, resposta):
    hist = [{'q': 'eu' if m.get('q') == 'eu' else 'bot', 't': str(m.get('t') or '')[:1500]}
            for m in (historico if isinstance(historico, list) else [])[-60:] if isinstance(m, dict)] + [{'q': 'bot', 't': resposta}]
    async with trava_leads:
        todos = ler_leads(); antigo = todos.get(sessao, {})
        lead = dict(antigo)
        lead.update({k: str(o.get(k) or antigo.get(k) or '').strip() for k in ('nome', 'contato', 'negocio', 'necessidade', 'resultado', 'prazo', 'solucao', 'resumo')})
        lead.update({'sessao': sessao, 'protocolo': antigo.get('protocolo') or protocolo(sessao), 'origem': pagina or 'site', 'historico': hist,
                     'data': antigo.get('data') or datetime.datetime.now(BRT).strftime('%d/%m/%Y %H:%M'),
                     'atualizado': datetime.datetime.now(BRT).strftime('%d/%m/%Y %H:%M'), 'atendente': bool(o.get('atendente') or antigo.get('atendente'))})
        avisar = (not antigo.get('avisado')) or (o.get('atendente') and not antigo.get('avisado_atendente'))
        if avisar:
            lead['avisado'] = True
            if o.get('atendente'): lead['avisado_atendente'] = True
        todos[sessao] = lead; gravar_leads(todos)
    log('lead %s salvo (%s)' % (lead['protocolo'], 'pediu atendente' if o.get('atendente') else 'pronto'))
    if avisar: asyncio.ensure_future(avisar_equipe(lead, bool(o.get('atendente'))))

# ---------------- aviso no WhatsApp da Murinoi (pela Evolution, rede interna) ----------------
_dono = ['', 0.0]
async def numero_dono():
    if _dono[0] and time.time() - _dono[1] < 3600: return _dono[0]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as s:
            async with s.get(NUM_FONTE) as r: pag = await r.text()
        k = re.search(r"const WA_K = '([^']+)'", pag)
        if k:   # o número fica só codificado no site; decodifica na hora, sem escrever em lugar nenhum
            _dono[0] = re.sub(r'\D', '', ''.join(chr(b ^ 23) for b in base64.b64decode(k.group(1)))[::-1]); _dono[1] = time.time()
    except Exception as e: log('aviso: não consegui ler o número (%s)' % e)
    return _dono[0]

async def avisar_equipe(d, atendente):
    destino = await numero_dono()
    kevo = os.environ.get('EVO_KEY', '').strip()   # chave da Evolution, passada pelo instalador
    if len(destino) < 12 or not kevo: return log('aviso no WhatsApp não enviado (sem número ou sem Evolution)')
    ou = lambda s, alt='-': (s or '').strip() or alt
    contato = ou(d.get('contato'), '')
    fone = re.sub(r'\D', '', contato)
    if len(fone) in (10, 11): fone = '55' + fone
    email = (re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', contato) or [''])[0]
    topo = '🙋 *Cliente pediu um atendente* (responder agora)' if atendente else '🔔 *Novo cliente pelo site*'
    msg = (topo + '\n' + d['protocolo'] + ' · ' + d['data'] + '\n\n👤 *' + ou(d.get('nome'), 'Não informado') + '*\n📞 ' + ou(contato) +
           '\n🏢 ' + ou(d.get('negocio')) + '\n\n🧩 *Precisa:* ' + ou(d.get('necessidade')) + '\n' +
           ('🎯 *Quer:* ' + d['resultado'] + '\n' if d.get('resultado') else '') + ('⏱️ *Prazo:* ' + d['prazo'] + '\n' if d.get('prazo') else '') +
           '💡 *Sugestão da Nina:* ' + ou(d.get('solucao')) + '\n\n' +
           ('👉 Responder: https://wa.me/' + fone if 12 <= len(fone) <= 13 else ('👉 Responder: ' + email if email else '')) +
           ('\n\n_Ficha completa: murinoi.com/nina/leads?c=' + COD + '_' if COD else ''))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as s:
            async with s.post(EVO_URL + '/message/sendText/murinoi', json={'number': destino, 'text': msg}, headers={'apikey': kevo}) as r:
                log('aviso no WhatsApp: %s' % ('enviado' if r.status < 300 else 'falhou (%s)' % r.status))
    except Exception as e: log('aviso no WhatsApp falhou: %s' % e)

# ---------------- rotas ----------------
async def chat(req):
    do_site(req); limite(req, 'chat', 20)
    try: b = await req.json()
    except Exception: raise web.HTTPBadRequest()
    sessao, msg = str(b.get('sessionId') or ''), str(b.get('message') or '')
    if not (8 <= len(sessao) <= 64) or not msg.strip() or len(msg) > 1500: raise web.HTTPBadRequest()
    try: o = await pensar_site(b.get('historico'), msg)
    except Exception as e:
        log('chat sem resposta: %s' % str(e)[:200]); raise web.HTTPServiceUnavailable()
    if o.get('etapa_humano') == 'pediu':
        o['resposta'] = 'Poderia explicar melhor, em poucas palavras ou de forma resumida, a sua solicitação?' + \
            ('' if o.get('contato') else ' Se puder, deixe também seu nome e WhatsApp com DDD para o atendente falar com você.')
    if o.get('etapa_humano') == 'explicou': o['resposta'] = 'Aguarde um momento que logo um dos nossos atendentes irá te responder.'
    o['atendente'] = o.get('etapa_humano') == 'explicou'
    pronto = o.get('pronto') is True
    if pronto or o['atendente']:
        try: await salvar_lead(sessao, o, b.get('historico'), str(b.get('page') or '')[:300], o['resposta'])
        except Exception as e: log('erro ao salvar lead: %s' % e)
    return web.json_response({'resposta': o['resposta'], 'pronto': pronto, 'atendente': o['atendente'],
                              'resumo': (o.get('resumo') or o.get('necessidade') or '') if (pronto or o['atendente']) else ''})

PEDIDO_TRANSCRICAO = ('Transcreva exatamente o que a pessoa fala neste áudio, em português do Brasil, palavra por palavra, na ordem em que foi dito. '
    'Não resuma, não traduza, não corrija o sentido, não repita trechos e não acrescente nada. Use pontuação, acentos e letras maiúsculas corretas. '
    'Contexto: a pessoa está conversando com a Nina, assistente da Murinoi (empresa de aplicativos, sites, sistemas, automações e inteligência artificial); '
    'nomes como Murinoi, Nina, WhatsApp, Pix e Google Play podem aparecer. Se não houver fala, responda exatamente: [silêncio]. Responda somente com o texto transcrito.')

def converter(dados):
    # deixa qualquer gravação (webm do Android/Chrome, mp4 do iPhone…) no formato que o Gemini lê melhor: ogg/opus mono 16 kHz
    with tempfile.TemporaryDirectory() as d:
        ent, sai = d + '/entrada', d + '/saida.ogg'
        open(ent, 'wb').write(dados)
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', ent, '-ac', '1', '-ar', '16000', '-c:a', 'libopus', '-b:a', '24k', '-t', '120', sai], timeout=60)
        if r.returncode != 0 or not os.path.exists(sai): raise ValueError('áudio ilegível')
        return open(sai, 'rb').read()

async def ouvir(req):
    do_site(req); limite(req, 'ouvir', 15)
    try: b = await req.json(); dados = base64.b64decode(str(b.get('audio') or ''), validate=False)
    except Exception: raise web.HTTPBadRequest()
    if not (300 < len(dados) < 3_000_000): raise web.HTTPBadRequest()
    try: ogg = await asyncio.get_running_loop().run_in_executor(None, converter, dados)
    except Exception: raise web.HTTPBadRequest(text='áudio ilegível')
    corpo = {'contents': [{'role': 'user', 'parts': [{'inline_data': {'mime_type': 'audio/ogg', 'data': base64.b64encode(ogg).decode()}}, {'text': PEDIDO_TRANSCRICAO}]}]}
    for modelo in MODELOS_OUVIR:
        try:
            t = texto_de(await gemini(modelo, dict(corpo, generationConfig=config_modelo(modelo, {'temperature': 0, 'maxOutputTokens': 2048})), 30))
            t = re.sub(r'^\s*(transcri[çc][ãa]o\s*:)\s*', '', t, flags=re.I).strip().strip('"“”').strip()
            if '[silêncio]' in t.lower() or '[silencio]' in t.lower(): t = ''
            return web.json_response({'texto': t}, headers={'Cache-Control': 'no-store'})
        except Exception as e:
            log('ouvir: %s falhou: %s' % (modelo, str(e)[:200]))
            if 'sem chave' in str(e): break
    raise web.HTTPServiceUnavailable()

CACHE_VOZ = DADOS + '/voz'; os.makedirs(CACHE_VOZ, exist_ok=True)
async def voz(req):
    do_site(req); limite(req, 'voz', 30)
    try: t = re.sub(r'\s+', ' ', str((await req.json()).get('texto') or '')).strip()
    except Exception: raise web.HTTPBadRequest()
    if not t or len(t) > 1200: raise web.HTTPBadRequest()
    arq = os.path.join(CACHE_VOZ, hashlib.sha1(t.encode()).hexdigest() + '.wav')
    cab = {'Content-Type': 'audio/wav', 'Cache-Control': 'public, max-age=86400'}
    if os.path.exists(arq): return web.FileResponse(arq, headers=cab)
    limpo = re.sub(r'[\U0001F300-\U0001FAFF☀-➿️]', '', t)
    corpo = {'contents': [{'parts': [{'text': 'Leia em português do Brasil, com voz feminina natural, calorosa e profissional, num ritmo tranquilo: ' + limpo}]}],
             'generationConfig': {'responseModalities': ['AUDIO'], 'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Kore'}}}}}
    try: d = await gemini(MODELO_VOZ, corpo, 40)
    except Exception as e: log('voz: %s' % str(e)[:200]); raise web.HTTPServiceUnavailable()
    parte = next((p for p in (((d.get('candidates') or [{}])[0].get('content') or {}).get('parts') or []) if p.get('inlineData')), None)
    if not parte: raise web.HTTPServiceUnavailable()
    pcm = base64.b64decode(parte['inlineData']['data'])
    taxa = int((re.search(r'rate=(\d+)', parte['inlineData'].get('mimeType', '')) or [0, 24000])[1])
    cab_wav = b'RIFF' + (36 + len(pcm)).to_bytes(4, 'little') + b'WAVEfmt ' + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') + \
        (1).to_bytes(2, 'little') + taxa.to_bytes(4, 'little') + (taxa * 2).to_bytes(4, 'little') + (2).to_bytes(2, 'little') + \
        (16).to_bytes(2, 'little') + b'data' + len(pcm).to_bytes(4, 'little')
    with open(arq + '.tmp', 'wb') as f: f.write(cab_wav + pcm)
    os.replace(arq + '.tmp', arq)
    fs = sorted(os.scandir(CACHE_VOZ), key=lambda e: e.stat().st_mtime)
    for e in fs[:max(0, len(fs) - 1500)]: os.remove(e.path)
    return web.FileResponse(arq, headers=cab)

# ---------------- chatbot do WhatsApp (Evolution) ----------------
SISTEMA_WHATS = """Você é a assistente virtual da Murinoi, empresa brasileira que cria soluções inteligentes para negócios e pessoas: aplicativos (inclusive na Google Play), sistemas, sites, automações e soluções com inteligência artificial. A promessa da Murinoi é simples: menos custo e mais tempo para o negócio do cliente.

Você atende pelo WhatsApp pessoas que chegaram pelo site murinoi.com. Seu objetivo é entender a necessidade, mostrar como a Murinoi pode ajudar e deixar tudo pronto para o especialista enviar a proposta.

Como conversar:
- Português do Brasil, tom cordial, leve e profissional. Trate por "você" e use o primeiro nome quando souber.
- Mensagens curtas, como no WhatsApp: no máximo 3 frases por resposta. Uma pergunta por vez.
- Use *negrito* do WhatsApp com moderação. Nada de títulos, tabelas ou listas longas. No máximo 1 emoji por mensagem.

Roteiro (adapte ao que a pessoa já contou, sem repetir perguntas):
1. Cumprimente, apresente-se como assistente da Murinoi e pergunte o que toma tempo ou dinheiro no negócio dela hoje.
2. Entenda o tipo de negócio e como a tarefa é feita hoje.
3. Explique em 1 ou 2 frases como a Murinoi resolveria (app, sistema, site, automação ou IA) e o ganho em tempo ou dinheiro.
4. Peça o nome (se ainda não souber) e o melhor horário para o especialista falar com ela.
5. Feche dizendo que o especialista da Murinoi vai enviar a solução, o prazo e o investimento por aqui mesmo, em breve.

Regras:
- Nunca invente preços, prazos fechados, clientes ou casos. Se perguntarem valor, diga que depende do escopo e que o especialista envia a proposta sem compromisso.
- Se a pessoa pedir para falar com uma pessoa, diga que o especialista da Murinoi já foi avisado e vai responder aqui em breve.
- Assuntos fora da Murinoi: responda com gentileza em uma frase e volte ao negócio da pessoa.
- Nunca peça senhas, dados de cartão ou documentos.
- Se perguntarem, o site é murinoi.com e o e-mail é contato@murinoi.com."""
memoria_whats = {}
async def bot(req):
    try: b = await req.json()
    except Exception: raise web.HTTPBadRequest()
    if b.get('instanceName') != 'murinoi' or not b.get('chatInput'): raise web.HTTPBadRequest()
    quem = str(b.get('remoteJid') or b.get('sessionId') or '')
    agora = time.time()
    for k in [k for k, v in memoria_whats.items() if agora - v['t'] > 6 * 3600]: memoria_whats.pop(k, None)
    mem = memoria_whats.setdefault(quem, {'t': agora, 'h': []}); mem['t'] = agora
    cab = 'Cliente: %s\nData e hora: %s\n\nMensagem:\n' % (b.get('pushName') or 'não informado', datetime.datetime.now(BRT).strftime('%d/%m/%Y %H:%M'))
    msg = str(b['chatInput'])[:2000]
    corpo = {'systemInstruction': {'parts': [{'text': SISTEMA_WHATS}]}, 'contents': conversa(mem['h'][-24:], msg, cab)}
    for modelo in MODELOS_CHAT:
        try:
            t = texto_de(await gemini(modelo, dict(corpo, generationConfig=config_modelo(modelo, {'temperature': 0.5, 'maxOutputTokens': 2048})), 40))
            if t:
                mem['h'] += [{'q': 'eu', 't': msg}, {'q': 'bot', 't': t}]
                return web.json_response({'output': t})
        except Exception as e: log('whatsapp: %s falhou: %s' % (modelo, str(e)[:200]))
    return web.json_response({'output': 'Recebi sua mensagem! Um especialista da Murinoi vai te responder aqui em breve.'})

# ---------------- páginas de administração (protegidas pelo código do log) ----------------
ESTILO = '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><style>body{margin:0;background:#f7f3ee;color:#2e2b33;font:16px/1.6 system-ui,-apple-system,sans-serif}main{max-width:760px;margin:0 auto;padding:28px 16px}h1{font-family:Georgia,serif;font-weight:400;font-size:1.9rem;margin:0 0 6px}p,li{color:#5b5662}.c{background:#fffdfa;border:1px solid #e4ddd6;border-radius:16px;padding:18px;margin:14px 0}input{width:100%;box-sizing:border-box;font:inherit;padding:12px 14px;border:1px solid #d9cfd6;border-radius:12px;background:#fff}button{font:500 1rem system-ui;border:0;border-radius:999px;padding:13px 24px;background:#2e2b33;color:#fff;margin-top:12px;cursor:pointer}.ok{color:#2f7a4f;font-weight:600}.erro{color:#a33;font-weight:600}b{color:#2e2b33}small{color:#8e8894}details{margin-top:8px}.m{padding:6px 10px;border-radius:10px;margin:4px 0;background:#f6f1f4}.eu{background:#e7e4f0}</style>'
def checar_codigo(req):
    if not COD or req.query.get('c') != COD: raise web.HTTPForbidden()

async def pagina_chave(req):
    checar_codigo(req); aviso = ''
    if req.method == 'POST':
        k = str((await req.post()).get('chave', '')).strip()
        try:
            await gemini(MODELOS_CHAT[0], {'contents': [{'role': 'user', 'parts': [{'text': 'Responda só: ok'}]}]}, 20, k=k)
            with open(ARQ_CHAVE + '.tmp', 'w') as f: f.write(k)
            os.chmod(ARQ_CHAVE + '.tmp', 0o600); os.replace(ARQ_CHAVE + '.tmp', ARQ_CHAVE)
            aviso = '<p class="ok">Chave salva e testada. A Nina já está respondendo, ouvindo e falando. ✅</p>'; log('chave do Gemini cadastrada e testada')
        except Exception as e:
            aviso = '<p class="erro">Essa chave não funcionou (%s). Confira se copiou a chave inteira.</p>' % html.escape(str(e)[:120])
    estado = '<p class="ok">A Nina já tem uma chave funcionando. Só troque se quiser usar outra.</p>' if chave() else '<p class="erro">A Nina ainda está sem chave: cadastre abaixo para ela voltar a responder.</p>'
    return web.Response(content_type='text/html', text='<!doctype html><html lang="pt-BR"><head><title>Chave da Nina</title>' + ESTILO + '</head><body><main><h1>Chave da Nina</h1>' + estado + aviso +
        '<div class="c"><ol><li>Abra <b>aistudio.google.com/apikey</b> com a conta Google da Murinoi.</li><li>Toque em <b>Create API key</b> (ou copie a chave que já existe).</li><li>Cole aqui e toque em Salvar. Ela fica guardada só na sua VPS.</li></ol>'
        '<form method="post"><input name="chave" type="password" autocomplete="off" placeholder="Cole a chave aqui (começa com AIza…)" required><button>Salvar e testar</button></form></div></main></body></html>')

async def pagina_leads(req):
    checar_codigo(req)
    e = html.escape
    itens = sorted(ler_leads().values(), key=lambda l: l.get('atualizado', ''), reverse=True)
    def data_ord(l):
        try: return datetime.datetime.strptime(l.get('atualizado', ''), '%d/%m/%Y %H:%M')
        except Exception: return datetime.datetime.min
    itens.sort(key=data_ord, reverse=True)
    cards = ''.join('<div class="c"><small>%s · %s%s</small><h3 style="margin:4px 0">%s</h3><p><b>Contato:</b> %s<br><b>Negócio:</b> %s<br><b>Precisa:</b> %s<br><b>Quer:</b> %s<br><b>Prazo:</b> %s</p><p><b>Sugestão da Nina:</b> %s</p><p><b>Resumo:</b><br>%s</p><details><summary>Conversa completa</summary>%s</details></div>' % (
        e(l.get('protocolo', '')), e(l.get('data', '')), ' · <b style="color:#a33">pediu atendente</b>' if l.get('atendente') else '', e(l.get('nome') or 'Sem nome'),
        e(l.get('contato') or '-'), e(l.get('negocio') or '-'), e(l.get('necessidade') or '-'), e(l.get('resultado') or '-'), e(l.get('prazo') or '-'),
        e(l.get('solucao') or '-'), e(l.get('resumo') or '-').replace('\n', '<br>'),
        ''.join('<div class="m%s">%s</div>' % (' eu' if m.get('q') == 'eu' else '', e(m.get('t', ''))) for m in l.get('historico', []))) for l in itens)
    return web.Response(content_type='text/html', text='<!doctype html><html lang="pt-BR"><head><title>Clientes da Nina</title>' + ESTILO + '</head><body><main><h1>Clientes atendidos pela Nina</h1><p>%d atendimento(s). Os dados são confidenciais (LGPD).</p>%s</main></body></html>' % (len(itens), cards or '<p>Nenhum ainda.</p>'))

async def saude(req):
    return web.json_response({'ok': True, 'ia': bool(chave())}, headers={'Cache-Control': 'no-store'})

app = web.Application(client_max_size=5 * 1024 * 1024)
app.router.add_post('/nina/chat', chat); app.router.add_post('/nina/ouvir', ouvir); app.router.add_post('/nina/voz', voz)
app.router.add_post('/nina/bot', bot); app.router.add_get('/nina/saude', saude)
app.router.add_route('*', '/nina/chave', pagina_chave); app.router.add_get('/nina/leads', pagina_leads)
if __name__ == '__main__':
    log('Nina no ar' + (' (com chave do Gemini)' if chave() else ' — falta cadastrar a chave do Gemini'))
    web.run_app(app, port=int(os.environ.get('PORTA', '8095')), access_log=None, print=None)
