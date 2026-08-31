import sys

files = {
    'README.md': [
        'Chitra.ai',
        'IBM AI Builders Challenge',
        'Problem Statement',
        'System Architecture',
        'Core Innovations',
        'Tech Stack',
        'Setup',
        'WATSONX_APIKEY',
        'WATSONX_PROJECT_ID',
        'python -m backend.app',
        'npm run dev',
        'api/v1/analyze-thermal',
        'Monte Carlo Dropout',
        'adjusted_confidence',
        'GraniteFallbackAgent',
        'ibm/granite-3-8b-instruct',
        'Space Exploration',
        'docs/ibm_bob_usage.md',
        'PyTorch',
        'YOLOv8',
        'FastAPI',
        'Next.js',
        'ChromaDB',
        'LangChain',
        'Tailwind CSS',
        'rasterio',
        'CUDA',
        'curl http://localhost:8000/health',
        'http://localhost:3000',
        'Responsible AI',
    ],
    'docs/ibm_bob_usage.md': [
        'IBM Bob',
        'Phase 0',
        'Phase 1',
        'Phase 2',
        'Phase 3',
        'Phase 4',
        'Phase 5',
        'Phase 6',
        'Phase 7',
        'MCDropout2d',
        'GraniteFallbackAgent',
        'run_mc_inference',
        'METRICS_JSON_START',
        '163 automated structural checks',
        'page.tsx',
        'ImageSlider.tsx',
        'app.py',
        'knowledge_base.py',
        'Bessel-corrected',
        'clamp(min=1e-8)',
        'isDragging',
        'inline SVG',
        'Read before writing',
        'read_file',
        'write_file',
        'update_todo_list',
    ],
}

all_ok = True
for path, required in files.items():
    src = open(path, encoding='utf-8').read()
    lines = src.count('\n')
    print(f'{path}  ({lines} lines, {len(src)} chars)')
    for s in required:
        ok = s in src
        if not ok:
            all_ok = False
        mark = 'OK  ' if ok else 'FAIL'
        print(f'  [{mark}]  {s}')
    print()

if not all_ok:
    sys.exit(1)
print('All documentation checks passed.')
