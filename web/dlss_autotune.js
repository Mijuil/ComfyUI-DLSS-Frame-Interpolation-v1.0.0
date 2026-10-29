/**
 * DLSS Frame Interpolation — "Analyze source" auto-tuning.
 *
 * Adds two entries to every DLSS node that owns the "Quality & cleanup" and
 * "Color & light" sections (the enable_enhance / enable_color toggles):
 *   - a canvas button "Analyze source" (classic canvas mode only)
 *   - context menu items "Analyze source (auto settings)" and
 *     "Reset enhancements to neutral" (both canvas modes)
 *
 * Clicking Analyze traces the node's media input upstream to the loader
 * (LoadImage / LoadVideo), asks the backend to measure the source and writes
 * the suggested values into the node's widgets. Touched widgets get an
 * "auto" label suffix that clears on manual edit. Reasons are logged to the
 * browser console.
 */
import { app } from '../../scripts/app.js'

const EXTENSION_NAME = 'DLSS.FrameInterpolation.AnalyzeSource.v2'

// Nodes that carry the enhancer/color sections.
const TARGET_WIDGETS = ['enable_enhance', 'enable_color']

// Loader node classes we can resolve, mapped to request kind + widget name.
const LOADERS = {
  LoadImage: { kind: 'image', widgets: ['image'] },
  LoadImageOutput: { kind: 'image', widgets: ['image'] },
  LoadVideo: { kind: 'video', widgets: ['file'] },
  VHS_LoadVideo: { kind: 'video', widgets: ['video', 'file'] },
  VHS_LoadVideoPath: { kind: 'video', widgets: ['video', 'file'] }
}

const AUTO_SUFFIX = ' · auto'

// Neutral values for the sections the analyzer controls.
const NEUTRAL = {
  enable_deband: false,
  edge_only: false,
  output_detail_strength: 1.0,
  gamma: 0.0,
  skin_smoothing: 0.0,
  clarity: 1.0,
  sharpen: 0.0,
  temporal_deflicker: 0.0,
  luma_match: 0.0,
  exposure: 0.0,
  contrast: 0.0,
  saturation: 0.0,
  shadows: 0.0,
  highlights: 0.0
}

function findSource (node) {
  // Walk the first media input slot upstream until a known loader appears.
  const inputNames = node.type === 'NvidiaDLSSImageUpscale' ? ['image'] : ['video']
  for (const slotName of inputNames) {
    const inputIndex = node.inputs?.findIndex((input) => input.name === slotName)
    if (inputIndex === undefined || inputIndex < 0) continue
    const linkId = node.inputs[inputIndex].link
    if (linkId == null) continue
    const link = app.graph.links[linkId]
    if (!link) continue
    const origin = app.graph._nodes_by_id[link.origin_id]
    if (!origin) continue
    const loader = LOADERS[origin.comfyClass || origin.type]
    if (!loader) continue
    for (const widgetName of loader.widgets) {
      const widget = origin.widgets?.find((w) => w.name === widgetName)
      if (widget && typeof widget.value === 'string' && widget.value) {
        return { kind: loader.kind, path: widget.value, loader: origin.comfyClass || origin.type }
      }
    }
  }
  return null
}

function markAuto (widget) {
  if (!widget.__dlssAutoLabel) widget.__dlssAutoLabel = widget.label || widget.name
  widget.label = widget.__dlssAutoLabel + AUTO_SUFFIX
  if (!widget.__dlssAutoClear) {
    widget.__dlssAutoClear = () => {
      if (widget.__dlssAutoLabel) widget.label = widget.__dlssAutoLabel
      widget.__dlssAutoLabel = null
    }
    const previousCallback = widget.callback
    widget.callback = function (...args) {
      widget.__dlssAutoClear()
      return previousCallback?.apply(this, args)
    }
  }
}

function applyValues (node, values, reasons, auto) {
  const touched = []
  for (const [name, raw] of Object.entries(values)) {
    const widget = node.widgets?.find((w) => w.name === name)
    if (!widget || widget.type === 'button') continue
    const isToggle = widget.type === 'toggle' || typeof widget.value === 'boolean'
    const next = isToggle ? !!raw : Number(raw)
    if (widget.value === next) continue
    if (auto) markAuto(widget)
    widget.value = next
    touched.push([name, next, reasons?.[name] || (auto ? '' : 'reset to neutral')])
  }
  if (touched.length) {
    node.setDirtyCanvas?.(true, true)
    app.graph.setDirtyCanvas?.(true, true)
  }
  return touched
}

function showToast (severity, summary, detail) {
  try {
    app.extensionManager?.toast?.add?.({ severity, summary, detail })
  } catch {
    console[severity === 'error' ? 'error' : 'log'](`[${summary}] ${detail}`)
  }
}

async function analyze (node) {
  const source = findSource(node)
  if (!source) {
    showToast('error', 'DLSS Analyze', 'Connect a LoadImage / LoadVideo node with a selected file first.')
    return
  }
  try {
    const response = await app.api.fetchApi('/dlss/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(source)
    })
    const payload = await response.json()
    if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`)
    const touched = applyValues(node, payload.values || {}, payload.reasons || {}, true)
    console.groupCollapsed(`[DLSS Analyze] ${node.title}: ${touched.length} setting(s) from ${source.loader}`)
    console.log('metrics', payload.metrics)
    for (const [name, value, reason] of touched) console.log(`${name} = ${value}  <- ${reason}`)
    console.groupEnd()
    showToast('success', 'DLSS Analyze',
      touched.length
        ? `Applied ${touched.length} setting(s) from ${source.loader}. Details in browser console (F12).`
        : 'Source looks neutral - nothing to change.')
  } catch (error) {
    console.error('[DLSS Analyze] failed:', error)
    showToast('error', 'DLSS Analyze', error.message)
  }
}

function resetToNeutral (node) {
  const touched = applyValues(node, NEUTRAL, {}, false)
  console.log(`[DLSS Analyze] ${node.title}: reset ${touched.length} setting(s) to neutral`)
  showToast('info', 'DLSS Analyze', `Reset ${touched.length} enhancement setting(s) to neutral.`)
}

app.registerExtension({
  name: EXTENSION_NAME,
  beforeRegisterNodeDef (nodeType, nodeData) {
    if (!TARGET_WIDGETS.some((name) =>
      name in (nodeData.input?.required ?? {}) || name in (nodeData.input?.optional ?? {}))) {
      return
    }

    const originalNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = originalNodeCreated?.apply(this, arguments)
      if (!this.__dlssAnalyzeButton) {
        try {
          const button = this.addWidget('button', 'Analyze source', null, () => analyze(this))
          button.serialize = false
          this.__dlssAnalyzeButton = button
        } catch {
          /* widget host unavailable - degrade silently */
        }
      }
      return result
    }

    // Context menu works in both the classic canvas and Vue Nodes 2.0.
    const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions
    nodeType.prototype.getExtraMenuOptions = function (canvas, menuOptions) {
      originalGetExtraMenuOptions?.apply(this, arguments)
      menuOptions.push(
        {
          content: 'Analyze source (auto settings)',
          callback: () => analyze(this)
        },
        {
          content: 'Reset enhancements to neutral',
          callback: () => resetToNeutral(this)
        }
      )
    }
  }
})
