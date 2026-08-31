import ast, sys, re

src  = open('backend/app.py', encoding='utf-8').read()
tree = ast.parse(src)
print('AST OK : backend/app.py')

classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
funcs   = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

required_funcs = {
    'lifespan',
    'health',
    'analyze_thermal',
    '_decode_thermal_image',
    '_decode_tiff',
    '_decode_pil_image',
    '_pad_to_multiple',
    '_array_to_base64_png',
    '_heatmap_to_base64_png',
    '_bbox_overlay_to_base64_png',
    '_cleanup_tensors',
    '_select_device',
}

missing_f = required_funcs - funcs
if missing_f:
    print('MISSING FUNCS:', missing_f)
    sys.exit(1)

print('Functions:', sorted(funcs))

checks = {
    'CORS middleware with localhost:3000':      'localhost:3000' in src,
    'lifespan startup caches generator':        'app.state.generator' in src,
    'lifespan startup caches detector':         'app.state.detector' in src,
    'lifespan startup caches agent':            'app.state.agent' in src,
    'POST /api/v1/analyze-thermal':             '/api/v1/analyze-thermal' in src,
    'file UploadFile parameter':                'UploadFile' in src,
    'mc_passes Form parameter':                 'mc_passes' in src,
    'latitude/longitude optional params':       'latitude' in src and 'longitude' in src,
    'normalize_thermal called':                 'normalize_thermal' in src,
    'run_mc_inference called':                  'run_mc_inference' in src,
    'tensor_to_uint8_rgb called':               'tensor_to_uint8_rgb' in src,
    'uncertainty_map_to_numpy called':          'uncertainty_map_to_numpy' in src,
    'detector.detect called':                   'detector.detect' in src,
    'agent.generate_intelligence_briefing':     'generate_intelligence_briefing' in src,
    'CUDA OOM handled (503)':                   'OutOfMemoryError' in src and '503' in src,
    'UnidentifiedImageError handled (422)':     'UnidentifiedImageError' in src and '422' in src,
    'torch.cuda.empty_cache in cleanup':        'torch.cuda.empty_cache' in src,
    'gc.collect in cleanup':                    'gc.collect' in src,
    'Base64 PNG encoded for rgb':               'reconstructed_rgb' in src,
    'Base64 PNG encoded for heatmap':           'uncertainty_heatmap' in src,
    'bbox_overlay encoded':                     'bbox_overlay' in src,
    'mean_uncertainty in response':             'mean_uncertainty' in src,
    'max_uncertainty in response':              'max_uncertainty' in src,
    'mc_passes_executed in response':           'mc_passes_executed' in src,
    'device in response':                       '"device"' in src,
    'agent_briefing in response':               'agent_briefing' in src,
    'pad_to_multiple of 16':                    'multiple=16' in src or 'multiple = 16' in src,
    'status success in response':               '"success"' in src,
    'GeoTIFF rasterio path':                    'rasterio' in src,
    'PIL grayscale fallback':                   'convert("L")' in src,
    'generator.eval() called at startup':       'generator.eval()' in src,
    'startup/shutdown lifespan used':           'lifespan' in src,
}

all_ok = True
for label, result in checks.items():
    mark = 'OK  ' if result else 'FAIL'
    if not result:
        all_ok = False
    print(f'  [{mark}] {label}')

if not all_ok:
    sys.exit(1)

print()
print('All checks passed.')
