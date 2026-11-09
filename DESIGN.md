# Blinkwatch visual identity

## 1. Atmosphere & Identity
Camera-first, precise, quiet. Blinkwatch is the violet camera-monitoring surface, distinct from Fucina's copper operations console, Finanza's petrol ledger and Agent Footprint's green environmental register. The signature is a violet event-count tile with segmented monitoring controls; snapshots and activity stay dominant, not a decorative hero.

## 2. Color
Use existing CSS tokens: `--bg`, `--surface`, `--surface-2`, `--ink`, `--muted`, `--line`. Violet `--accent` is #6d28d9 in light and #c4b5fd in dark; `--accent-soft` is #f1eafe / #302044; `--on-accent` provides inverse foreground. Neutral surfaces dominate; accent is reserved for navigation, active controls and the leading event metric. Existing green/amber/red classification and health meanings remain unchanged. Text contrast target WCAG AA; never replace semantic warnings with brand violet.

## 3. Typography
Use existing Geist / system sans and Geist Mono / system monospace. Body inherits the 1.5 line height; labels 12–13px, controls 13–14px, metrics 24–32px with tabular numerals, page titles 26–32px. Weight 500–600 distinguishes values from labels; no third font or external font dependency added by this change.

## 4. Spacing & Layout
4px base; 4/8/12/16/24/32px spacing. Existing shell max-width 1200px, 16px side gutters on mobile. Page scroll owns content; segmented control rows may scroll horizontally within their own bounded container. Preserve the floating bottom dock and page bottom padding. Support 320px upward without document overflow.

## 5. Components
Leading event stat: quiet label, dominant violet value on a soft violet tile. Remaining stats stay neutral. Category and labelling button groups use an inset neutral track with 4px padding, a surface-colored active segment, and violet selected text; native button handlers and existing `aria-pressed` or active classes are retained. Label groups have named `role=group`. Hover uses a neutral raised surface; focus-visible sits above adjacent segments with a 2px accent outline. Existing disabled states remain inert with reduced opacity. Loading, error and empty-state elements/handlers are unchanged. Count labels use a neutral compact outlined badge rather than looking like an action.

## 6. Motion & Interaction
Keep existing .15s easing for color/border changes. No new animations. Reduced motion removes transitions and animations; keyboard access uses native buttons and existing tab order. Do not synthesize toggle semantics that existing JavaScript does not keep in sync.

## 7. Depth & Surface
Content cards remain opaque with hairline borders and existing small neutral shadows; only sticky controls and navigation float. Segmented tracks are inset neutral surfaces, not additional glass layers. No gradients, decorative glow or new blur.

## 8. Accessibility Constraints & Accepted Debt
Preserve labels, handlers, IDs, routes, semantic alerts, data relationships and 44px touch targets. Focus outline must not be covered by neighboring controls. Maintain forced-color support. Existing external font loading and camera/inference workflows are unchanged; this is a UI pass, not a full accessibility or backend audit.

## 9. PanelUI provenance
Concrete references consulted: https://panelui.dev/r/web/stat.json and https://panelui.dev/r/web/button-group.json . Adapt the label/value anatomy, inset segmented track, named native group, and raised focus ordering to existing HTML/CSS. This is an original vanilla implementation of those patterns, not copied React/Tailwind source or an installed PanelUI package. No framework migration.

## Safe deployment
The web service shares the `blinkwatch` image with watcher and ipcam. Do not run compose-wide up/recreate for UI-only edits. Sync `blinkwatch/index.html`, `cameras.html`, `label.html`, `power.html`, and `liquid-glass.css` into `/app/blinkwatch` in `blinkpy_test-web-1` only. Back up host and container files first. HTML CSS cache version: `20261104-violet-2`. Restart only the web container if needed; never restart/recreate watcher or ipcam. Persist changes in this repository so the next intentional image build carries them forward; a web-container recreation without rebuilding from updated source would revert the running copy.
