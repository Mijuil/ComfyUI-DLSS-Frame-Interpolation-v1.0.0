/**
 * DLSS Frame Interpolation — one-click "Analyze source" auto-tuning.
 *
 * Adds an "Analyze source" button to every DLSS node that owns the
 * "Quality & cleanup" and "Color & light" sections (enable_enhance /
 * enable_color toggles). Context menu carries the same action plus a
 * "Reset enhancements to neutral" fallback.
 *
 * One click: scattered manual values are first returned to the node's own
 * defaults (read from the Python schema, so gamut like gamma 0.25..2.5 can
 * never be violated), then the backend measures the connected source
 * (image, or an even sample of video frames) and the optimal values are
 * written into the widgets. Every changed widget gets an "auto" label
 * suffix; touching that widget manually clears the mark. Manual
 * fine-tuning is never blocked.
 *
 * Feedback is shown on the button itself and as a toast - no devtools needed.
 */
import { app } from '../../scripts/app.js'

const EXTENSION_NAME = 'DLSS.FrameInterpolation.AnalyzeSource.v4'
const BUTTON_LABEL = 'Analyze source'

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

// Enhancement widgets the one-click reset covers. Values are NOT hardcoded:
// the neutral state is read from the node schema defaults at registration.
const TUNABLE = [
  'enable_enhance', 'enable_deband', 'gamma', 'skin_smoothing',
  'clarity', 'sharpen', 'edge_only', 'temporal_deflicker',
  'output_detail_strength',
  'enable_color', 'luma_match', 'exposure', 'contrast',
  'saturation', 'shadows', 'highlights'
]

function buildNeutral (nodeData) {
  const neutral = {}
  for (const section of [nodeData.input?.required, nodeData.input?.optional]) {
    for (const name of TUNABLE) {
      const def = section?.[name]
      if (Array.isArray(def) && def[1] && 'default' in def[1]) neutral[name] = def[1].default
    }
  }
  return neutral
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
    let next = isToggle ? !!raw : Number(raw)
    // Respect the widget's own bounds so a stale backend or schema can
    // never push a slider out of range (ComfyUI rejects out-of-range input).
    if (!isToggle) {
      const min = widget.options?.min
      const max = widget.options?.max
      if (Number.isFinite(min)) next = Math.max(min, next)
      if (Number.isFinite(max)) next = Math.min(max, next)
    }
    if (widget.value === next) continue
    if (auto) markAuto(widget)
    widget.value = next
    touched.push([name, next, reasons?.[name] || ''])
  }
  if (touched.length) {
    node.setDirtyCanvas?.(true, true)
    app.canvas?.setDirtyCanvas?.(true, true)
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

function flashButton (node, label, revertMs) {
  const button = node.__dlssAnalyzeButton
  if (!button) return
  button.name = label
  node.setDirtyCanvas?.(true, true)
  setTimeout(() => {
    button.name = BUTTON_LABEL
    node.setDirtyCanvas?.(true, true)
  }, revertMs)
}

async function analyze (node) {
  const source = findSource(node)
  if (!source) {
    flashButton(node, '✗ Connect a source first', 5000)
    showToast('error', 'DLSS Analyze', 'Connect a LoadImage / LoadVideo node with a selected file, then run Analyze again.')
    return
  }
  flashButton(node, 'Analyzing…', 30000)
  try {
    const response = await app.api.fetchApi('/dlss/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(source)
    })
    const payload = await response.json()
    if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`)
    // Node-default baseline first, analyzer suggestions on top - one click
    // always ends in a coherent state, never in a mix of old manual values.
    const neutral = node.__dlssNeutral ?? {}
    const merged = { ...neutral, ...(payload.values || {}) }
    const reasons = { ...payload.reasons }
    for (const key of Object.keys(neutral)) {
      if (!(key in (payload.values || {}))) reasons[key] = 'reset to default'
    }
    const touched = applyValues(node, merged, reasons, true)
    console.groupCollapsed(`[DLSS Analyze] ${node.title}: ${touched.length} setting(s) from ${source.loader}`)
    console.log('metrics', payload.metrics)
    for (const [name, value, reason] of touched) console.log(`${name} = ${value}  <- ${reason}`)
    console.groupEnd()
    if (touched.length) {
      flashButton(node, `✓ Applied ${touched.length} setting(s)`, 6000)
      showToast('success', 'DLSS Analyze',
        `${touched.length} setting(s) tuned to the source (marked "· auto"). Tweak any slider freely - the mark clears where you edit.`)
    } else {
      flashButton(node, '✓ Source is clean, settings neutral', 6000)
      showToast('info', 'DLSS Analyze', 'The source looks clean - enhancements set to neutral.')
    }
  } catch (error) {
    console.error('[DLSS Analyze] failed:', error)
    flashButton(node, '✗ Analyze failed', 6000)
    showToast('error', 'DLSS Analyze', error.message)
  }
}

function resetToNeutral (node) {
  const touched = applyValues(node, node.__dlssNeutral ?? {}, {}, false)
  showToast('info', 'DLSS Analyze', `Reset ${touched.length} enhancement setting(s) to default.`)
}

app.registerExtension({
  name: EXTENSION_NAME,
  beforeRegisterNodeDef (nodeType, nodeData) {
    if (!TARGET_WIDGETS.some((name) =>
      name in (nodeData.input?.required ?? {}) || name in (nodeData.input?.optional ?? {}))) {
      return
    }
    const neutral = buildNeutral(nodeData)

    const originalNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = originalNodeCreated?.apply(this, arguments)
      this.__dlssNeutral = { ...neutral }
      if (!this.__dlssAnalyzeButton) {
        try {
          const button = this.addWidget('button', BUTTON_LABEL, null, () => analyze(this))
          button.serialize = false
          this.__dlssAnalyzeButton = button
        } catch {
          /* widget host unavailable - menu entry still works */
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
          content: `⭐ ${BUTTON_LABEL} (auto settings)`,
          callback: () => analyze(this)
        },
        {
          content: 'Reset enhancements to default',
          callback: () => resetToNeutral(this)
        }
      )
    }
  }
})
