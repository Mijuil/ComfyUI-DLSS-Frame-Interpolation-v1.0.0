/**
 * DLSS Frame Interpolation — "Analyze source" button.
 *
 * Adds a button to every DLSS node that owns the "Quality & cleanup" and
 * "Color & light" sections (the enable_enhance / enable_color toggles).
 * Clicking it traces the node's media input upstream to the loader
 * (LoadImage / LoadVideo), asks the backend to measure the source and
 * writes the suggested values into the node's widgets.
 *
 * Every widget touched by the analyzer gets an "auto" label suffix so the
 * user sees exactly what was changed; the suffix clears as soon as the
 * user edits that widget manually. Reasons are logged to the browser
 * console. Vue Nodes 2.0 degrades silently (no canvas widgets there).
 */
import { app } from '../../scripts/app.js'

const EXTENSION_NAME = 'DLSS.FrameInterpolation.AnalyzeSource'

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

function isVueNodesMode () {
  try {
    return app.ui?.settings?.getSettingValue?.('Comfy.VueNodes.Enabled') === true
  } catch {
    return false
  }
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

function applySuggestions (node, values, reasons) {
  const touched = []
  for (const [name, raw] of Object.entries(values)) {
    const widget = node.widgets?.find((w) => w.name === name)
    if (!widget || widget.type === 'button') continue
    const isToggle = widget.type === 'toggle' || typeof widget.value === 'boolean'
    const next = isToggle ? !!raw : Number(raw)
    if (widget.value === next) continue
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
    widget.value = next
    widget.__dlssAutoClearReason = reasons?.[name] || ''
    touched.push([name, next, widget.__dlssAutoClearReason])
  }
  if (touched.length) node.setDirtyCanvas?.(true, true)
  return touched
}

async function analyze (node) {
  const source = findSource(node)
  if (!source) {
    app.ui?.dialog?.show?.(
      'DLSS Analyze: connect a LoadImage / LoadVideo node to this node first.',
      'error'
    ) ?? console.warn('[DLSS Analyze] no connected loader found')
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
    const touched = applySuggestions(node, payload.values || {}, payload.reasons || {})
    console.groupCollapsed(`[DLSS Analyze] ${node.title}: ${touched.length} setting(s) from ${source.loader}`)
    console.log('metrics', payload.metrics)
    for (const [name, value, reason] of touched) console.log(`${name} = ${value}  <- ${reason}`)
    console.groupEnd()
    if (!touched.length) console.log('[DLSS Analyze] source looks neutral - nothing to change')
  } catch (error) {
    console.error('[DLSS Analyze] failed:', error)
    app.ui?.dialog?.show?.(`DLSS Analyze failed: ${error.message}`, 'error')
      ?? app.extensionManager?.toast?.add?.({ severity: 'error', summary: 'DLSS Analyze', detail: error.message })
  }
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
      if (!isVueNodesMode() && !this.__dlssAnalyzeButton) {
        try {
          const button = this.addWidget('button', 'Analyze source', null, () => analyze(this))
          button.serialize = false
          this.__dlssAnalyzeButton = button
          // Keep the button as the very first row of the node.
          const index = this.widgets.indexOf(button)
          if (index > 0) this.widgets.splice(index, 1)
          this.widgets.unshift(button)
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
      menuOptions.push({
        content: 'Analyze source (auto settings)',
        callback: () => analyze(this)
      })
    }
  }
})
