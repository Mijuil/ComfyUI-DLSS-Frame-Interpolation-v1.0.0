/**
 * DLSS Frame Interpolation — canvas section headers.
 *
 * Python layer already exposes section titles as `display_name` of the four
 * section-toggle inputs (`enable_nr`, `output_detail_strength`,
 * `enable_enhance`, `enable_color`), so they are always visible as widget
 * labels (panel + Vue nodes fallback). This extension adds the canvas-only
 * cosmetics on top:
 *   - reserves a banner strip above each section toggle (computeSize wrap);
 *   - shifts the toggle's rendering/hit area to the strip bottom
 *     (public `y` field interception — widget input slot circles follow via
 *     LGraphNode._arrangeWidgetInputSlots, which reads `widget.y`);
 *   - draws the section title as a banner in the reserved strip
 *     (onDrawBackground runs before drawWidgets);
 *   - clips the toggle's top margin sliver so the widget shape does not
 *     bleed into the banner row (the label text itself stays visible).
 *
 * Only `../../scripts/app.js` is imported — the frontend bundle does not
 * ship a standalone litegraph module, so `LiteGraph` must not be imported
 * here. Everything degrades silently: Vue Nodes 2.0 never calls canvas
 * hooks and keeps plain display_name labels.
 */
import { app } from '../../scripts/app.js'

const EXTENSION_NAME = 'DLSS.FrameInterpolation.SectionHeaders'

const SECTION_BANNER_HEIGHT = 18
const BANNER_GAP = 4
const STRIP = SECTION_BANNER_HEIGHT + BANNER_GAP

// Fallback row height for widgets that somehow lack computeSize().
// Matches litegraph's default NODE_WIDGET_HEIGHT.
const FALLBACK_WIDGET_HEIGHT = 20

const SECTIONS = {
  enable_nr: null,
  output_detail_strength: null,
  enable_enhance: null,
  enable_color: null
}
const SECTION_NAMES = Object.keys(SECTIONS)

const SECTION_BG = 'rgba(20, 24, 34, 0.65)'
const SECTION_ACCENT = '#4f8cc9'
const SECTION_TEXT = '#9db8d4'

function isVueNodesMode () {
  // Mirror of LiteGraph.vueNodesMode without importing litegraph:
  // the frontend derives that flag from the same setting.
  try {
    return app.ui?.settings?.getSettingValue?.('Comfy.VueNodes.Enabled') === true
  } catch {
    return false
  }
}

function wrapWidgetY (widget) {
  // The banner strip occupies the top of the widget's computed height;
  // the interactive widget itself lives at the bottom of that block.
  let real = 0
  Object.defineProperty(widget, 'y', {
    configurable: true,
    get () {
      return real + STRIP
    },
    set (v) {
      real = v
    }
  })
}

function wrapComputeSize (widget) {
  const original = widget.computeSize
  widget.computeSize = function (width) {
    const size = original
      ? original.call(this, width)
      : [width, FALLBACK_WIDGET_HEIGHT]
    return [size[0], size[1] + STRIP]
  }
}

app.registerExtension({
  name: EXTENSION_NAME,
  beforeRegisterNodeDef (nodeType, nodeData) {
    if (!SECTION_NAMES.some((name) => name in (nodeData.input?.required ?? {}) ||
      name in (nodeData.input?.optional ?? {}))) {
      return
    }

    nodeType.prototype.onDrawBackground = function (ctx) {
      // drawNodeShape has already rendered the body: its local origin is the
      // widget area (below the title bar).
      if (this.flags?.collapsed || isVueNodesMode()) return

      ctx.save()
      for (const widget of this.widgets ?? []) {
        if (!(widget.name in SECTIONS)) continue
        const title = widget.label || widget.name
        const top = widget.y - STRIP
        const width = this.size[0]

        ctx.fillStyle = SECTION_BG
        ctx.beginPath()
        ctx.roundRect(6, top, width - 12, SECTION_BANNER_HEIGHT, 3)
        ctx.fill()

        ctx.fillStyle = SECTION_ACCENT
        ctx.fillRect(6, top, 2, SECTION_BANNER_HEIGHT)

        ctx.fillStyle = SECTION_TEXT
        ctx.font = '600 10px Arial, sans-serif'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'middle'
        ctx.fillText(title.toUpperCase(), 13, top + SECTION_BANNER_HEIGHT * 0.5)
      }
      ctx.restore()
    }

    const originalNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = originalNodeCreated?.apply(this, arguments)
      for (const widget of this.widgets ?? []) {
        if (!(widget.name in SECTIONS) || widget.__dlssSection) continue
        widget.__dlssSection = true
        wrapComputeSize(widget)
        wrapWidgetY(widget)
        if (typeof widget.drawWidget === 'function') {
          const drawWidget = widget.drawWidget
          widget.drawWidget = function (ctx, options) {
            // Clip the toggle's top margin sliver so the widget shape does
            // not bleed into the banner row (the label text stays visible).
            ctx.save()
            ctx.beginPath()
            ctx.rect(0, this.y, Number.MAX_SAFE_INTEGER, this.height)
            ctx.clip()
            drawWidget.call(this, ctx, options)
            ctx.restore()
          }
        }
      }
      return result
    }
  }
})
