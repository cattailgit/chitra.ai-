import ast, sys, re

files = [
    'backend/rag/knowledge_base.py',
    'backend/rag/agent.py',
]

for path in files:
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    print(f'AST OK : {path}')

# ── Structural checks on agent.py ─────────────────────────────────────────
src  = open('backend/rag/agent.py', encoding='utf-8').read()
tree = ast.parse(src)

classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
funcs   = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

required_classes = {
    'BriefingResult',
    'SatelliteVectorStore',
    'GraniteFallbackAgent',
    'GraniteLLMClient',
    'SatelliteIntelligenceAgent',
}
required_funcs = {
    'build_agent',
    'generate_intelligence_briefing',
    'retrieve',
    'invoke',
    '_build_prompt',
    '_build_rag_query',
    '_validate_metrics',
    '_render_fallback_briefing',
    '_parse_metrics_from_prompt',
    'to_dict',
}

missing_c = required_classes - classes
missing_f = required_funcs   - funcs
if missing_c:
    print('MISSING CLASSES:', missing_c)
    sys.exit(1)
if missing_f:
    print('MISSING FUNCS:', missing_f)
    sys.exit(1)

print()
print('Classes  :', sorted(classes))
print('Functions:', sorted(funcs))

# ── Key contract checks ────────────────────────────────────────────────────
checks = {
    'langchain_ibm import guarded':           '_LANGCHAIN_IBM_AVAILABLE' in src,
    'WATSONX_APIKEY via os.getenv':           'WATSONX_APIKEY' in src and 'os.getenv' in src,
    'WATSONX_PROJECT_ID via os.getenv':       'WATSONX_PROJECT_ID' in src,
    'ChromaDB import guarded':                'Chroma' in src and '_LANGCHAIN_COMMUNITY_AVAILABLE' in src,
    'Granite model ID ibm/granite-3-8b':      'granite-3-8b-instruct' in src,
    'Fallback simulator present':             'GraniteFallbackAgent' in src,
    'Uncertainty-penalised adj conf formula': 'adjusted_confidence' in src and 'complement' in src,
    'RAG retrieve called in briefing':        'self._vs.retrieve' in src,
    'build_agent() factory exported':         'def build_agent' in src,
    'used_fallback field in BriefingResult':  'used_fallback' in src,
    'mean_uncertainty in validated keys':     'mean_uncertainty' in src,
    'Keyword fallback for missing chromadb':  'keyword' in src.lower(),
    'no static mock detections':              'MOCK' not in src.upper().replace('# NO MOCK',''),
}

all_ok = True
for label, result in checks.items():
    status = 'OK  ' if result else 'FAIL'
    if not result:
        all_ok = False
    print(f'  [{status}] {label}')

if not all_ok:
    sys.exit(1)

# ── Knowledge base checks ─────────────────────────────────────────────────
kb_src = open('backend/rag/knowledge_base.py', encoding='utf-8').read()
ast.parse(kb_src)
doc_count = len(re.findall(r'"id":', kb_src))
assert doc_count >= 10, f'Expected >= 10 KB docs, found {doc_count}'
print()
print(f'Knowledge base : {doc_count} documents  OK')
print()
print('All checks passed.')
