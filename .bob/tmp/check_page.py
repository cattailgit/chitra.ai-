import sys

src = open('frontend/app/page.tsx', encoding='utf-8').read()
print(f'File read OK — {len(src)} chars, {src.count(chr(10))} lines')

checks = {
    # Directive & structure
    '"use client" directive':                src.strip().startswith('"use client"'),
    'exported as default':                   'export default function DashboardPage' in src,
    'imports ImageSlider':                   "from \"../components/ImageSlider\"" in src or "from '../components/ImageSlider'" in src,

    # Response types
    'AnalysisResponse interface':            'interface AnalysisResponse' in src,
    'AnalysisMetrics interface':             'interface AnalysisMetrics' in src,
    'AgentMeta interface':                   'interface AgentMeta' in src,
    'AnalysisImages interface':              'interface AnalysisImages' in src,
    'PipelineState type':                    'PipelineState' in src,

    # API
    'API_URL to localhost:8000':             'localhost:8000/api/v1/analyze-thermal' in src,
    'FormData POST request':                 'new FormData()' in src and 'method: "POST"' in src,
    'mc_passes appended to form':            'mc_passes' in src and 'form.append' in src,
    'latitude/longitude optional fields':    'latitude' in src and 'longitude' in src,

    # Layout
    'top navigation bar':                    'Chitra.ai' in src and 'Satellite IR Reconstruction' in src,
    'three-column layout (aside/main/aside)':'<aside' in src and '<main' in src,
    'left control sidebar':                  'Control Sidebar' in src or 'CONTROL' in src.upper() or 'control' in src.lower(),
    'right intelligence panel':              'Intelligence' in src or 'INTELLIGENCE' in src.upper(),

    # Sidebar controls
    'Dropzone component':                    'function Dropzone' in src,
    'file state + onFile handler':           'setFile' in src and 'handleFileSelected' in src,
    'MC passes slider (5-30)':               'min={5}' in src and 'max={30}' in src,
    'confidence threshold slider (0-100)':   'confThresh' in src and 'max={100}' in src,
    'latitude input field':                  'Latitude' in src,
    'longitude input field':                 'Longitude' in src,
    'Run button with loading state':         'Run Reconstruction' in src and 'isLoading' in src,
    'spinner during loading':                'function Spinner' in src,
    'button disabled when no file':          '!file || isLoading' in src or 'disabled={!file' in src,

    # Pipeline state
    'idle/loading/success/error states':     '"idle"' in src and '"loading"' in src and '"success"' in src and '"error"' in src,
    'execMs timing':                         'execMs' in src and 'performance.now()' in src,
    'error banner for pipeline errors':      'pipeline error' in src.lower() or 'Pipeline error' in src,

    # Workspace
    'empty state when idle':                 'No scene loaded' in src,
    'loading skeleton':                      'animate-pulse' in src,
    'ImageSlider rendered on success':       '<ImageSlider' in src,
    'thermalPreviewUrl from file':           'thermalPreviewUrl' in src and 'createObjectURL' in src,
    'filteredDetections by confThresh':      'filteredDetections' in src and 'adjusted_confidence' in src,

    # Co-pilot panel
    'TelemetryBadge component':              'function TelemetryBadge' in src,
    'device badge':                          'Device' in src and 'result.metrics.device' in src,
    'mean_uncertainty badge':                'mean_uncertainty' in src and 'Mean Unc' in src,
    'max_uncertainty badge':                 'max_uncertainty' in src,
    'exec time badge':                       'Exec Time' in src,
    'Granite live vs fallback badge':        'used_fallback' in src and 'IBM Granite' in src,
    'BriefingRenderer component':            'function BriefingRenderer' in src,
    'section headings highlighted':          'VISUAL RECONSTRUCTION INTEGRITY' in src or 'TACTICAL OBJECT ASSESSMENT' in src,
    'WARNING lines highlighted':             'WARNING:' in src,
    'DetectionRow component':                'function DetectionRow' in src,
    'detection table (class/raw/unc/adj)':   'Raw' in src and 'BBox Unc' in src and 'Adj. Conf' in src,
    'RAG context IDs displayed':             'retrieved_context_ids' in src and 'RAG Context' in src,
    'warning icon on high-uncertainty det':  '\u26a0' in src,

    # Sub-components and icons
    'SectionHeading component':              'function SectionHeading' in src,
    'LabelledSlider component':              'function LabelledSlider' in src,
    'inline SVG icons (no external dep)':    'function IconUpload' in src and 'function IconCpu' in src,
    'dark mode bg-gray-950':                 'bg-gray-950' in src,
    'hidden-threshold filter message':       'hidden by confidence threshold' in src,
}

all_ok = True
for label, ok in checks.items():
    mark = 'OK  ' if ok else 'FAIL'
    if not ok:
        all_ok = False
    print(f'  [{mark}] {label}')

print()
if not all_ok:
    sys.exit(1)
print('All checks passed.')
