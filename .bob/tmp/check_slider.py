import sys

src = open('frontend/components/ImageSlider.tsx', encoding='utf-8').read()
print('File read OK — size:', len(src), 'chars')

checks = {
    'use client directive':                src.strip().startswith('"use client"'),
    'ImageSliderProps interface':          'interface ImageSliderProps' in src,
    'Detection interface with bbox':       'interface Detection' in src and 'bbox: BBox' in src,
    'BBox interface (x1/y1/x2/y2)':       'interface BBox' in src and 'x1:' in src,
    'thermalSrc prop':                     'thermalSrc' in src,
    'reconstructedRgbSrc prop':            'reconstructedRgbSrc' in src,
    'uncertaintyHeatmapSrc prop':          'uncertaintyHeatmapSrc' in src,
    'bboxOverlaySrc prop':                 'bboxOverlaySrc' in src,
    'detections prop':                     'detections: Detection[]' in src,
    'imageWidth / imageHeight props':      'imageWidth' in src and 'imageHeight' in src,
    'splitFraction state (drag divider)':  'splitFraction' in src and 'setSplitFraction' in src,
    'isDragging ref':                      'isDragging' in src,
    'Heatmap toggle state':                'showHeatmap' in src and 'setShowHeatmap' in src,
    'Heatmap alpha/opacity slider':        'heatmapAlpha' in src,
    'Bbox toggle state':                   'showBboxes' in src,
    'Canvas ref + drawImage calls':        'canvasRef' in src and 'drawImage' in src,
    'scaleX / scaleY for bbox mapping':    'scaleX' in src and 'scaleY' in src,
    'hitTest hover function':              'hitTest' in src,
    'hoverPos state (x,y)':               'hoverPos' in src,
    'hoveredDet state':                    'hoveredDet' in src,
    'HoverTooltip sub-component':          'function HoverTooltip' in src,
    'Tooltip shows pixel X/Y':             'X:' in src and 'Y:' in src,
    'Tooltip shows raw_confidence':        'raw_confidence' in src,
    'Tooltip shows mean_uncertainty':      'mean_uncertainty' in src,
    'Tooltip shows adjusted_confidence':   'adjusted_confidence' in src,
    'ToggleSwitch sub-component':          'function ToggleSwitch' in src,
    'DetectionList sub-component':         'function DetectionList' in src,
    'LegendDot sub-component':             'function LegendDot' in src,
    'Touch support (onTouchStart)':        'onTouchStart' in src and 'touchmove' in src,
    'ResizeObserver for responsive scale': 'ResizeObserver' in src,
    'aspect ratio preserved':              'aspectRatio' in src,
    'Divider handle circle drawn':         'arc(' in src,
    'Divider arrows drawn':                'arrowOffset' in src,
    'Panel labels (THERMAL / RGB)':        'THERMAL INPUT' in src and 'RECONSTRUCTED RGB' in src,
    'Confidence colour tiers (3)':         '0.6' in src and '0.3' in src and '#00c83c' in src,
    'detectionColour helper':              'function detectionColour' in src,
    'Warning icon for high uncertainty':   '\u26a0' in src,
    '_drawBboxes fallback helper':         'function _drawBboxes' in src,
    '_drawLabel helper':                   'function _drawLabel' in src,
    'useImage custom hook':                'function useImage' in src,
    'useContainerWidth custom hook':       'function useContainerWidth' in src,
    'useEffect render loop':               'useEffect' in src,
    'useCallback for handlers':            'useCallback' in src,
    'exported as default':                 'export default function ImageSlider' in src,
}

all_ok = True
for label, result in checks.items():
    mark = 'OK  ' if result else 'FAIL'
    if not result:
        all_ok = False
    print(f'  [{mark}] {label}')

print()
print(f'Line count : {src.count(chr(10))}')
if not all_ok:
    sys.exit(1)
print()
print('All checks passed.')
