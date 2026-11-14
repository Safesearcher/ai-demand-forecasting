# AI Demand Forecasting and Inventory Decision Support

A high-performance, award-level web application for probabilistic retail demand forecasting and automated inventory decision support, built strictly to the specifications of `AI_Demand_Forecasting_SRS_v3_short.md`.

---

## 1. Product Overview

Traditional retail planning relies on single-point heuristics that hide variance and lead directly to costly stockouts or bloated carrying costs. **AI Demand Forecasting and Inventory Decision Support** provides an end-to-end operational decision cockpit that pairs machine-learned probabilistic quantile forecasts ($P_{10}$, $P_{50}$, $P_{90}$) with standard periodic-review $(R, S)$ inventory safety stock calculations.

### Core Mathematical Engine (SRS Section 4)
The application computes optimal replenishment quantities $Q$ and evaluates stockout vulnerabilities in real time:

$$\text{Protection Interval: } T = L + R$$
$$\text{Safety Stock: } SS = z \times \sigma_d \times \sqrt{T}$$
$$\text{Recommended Order Quantity: } Q = \max\left(0, \left\lceil D_T + SS - S - I \right\rceil\right)$$
$$\text{Stockout Risk Alert: flagged if } (S + I) < P_{90} \text{ lead-time demand over } L$$

#### Verified Worked Example
- **Parameters**: Lead time $L = 3\text{ d}$, Review period $R = 7\text{ d}$ ($T = 10\text{ d}$), Demand over interval $D_T = 120\text{ units}$, Daily standard deviation $\sigma_d = 5.7\text{ units/day}$, Service level factor $z = 1.65$ (95% non-stockout probability), Stock on hand $S = 40\text{ units}$, Incoming inventory $I = 0\text{ units}$.
- **Result**:
  $$SS = 1.65 \times 5.7 \times \sqrt{10} \approx 29.74 \approx 30\text{ units}$$
  $$Q = \max(0, \lceil 120 + 29.74 - 40 - 0 \rceil) = \lceil 109.74 \rceil = 110\text{ units}$$

---

## 2. Architecture & Design Direction

Built using a **zero-build vanilla architecture** (pure HTML5, CSS3, and modern ES JavaScript) that runs natively both via direct file opening (`file:///`) and local HTTP servers.

- **Routing**: Client-side hash routing (`#/dashboard`, `#/item`, `#/evaluation`, `#/upload`, `#/models`) with View Transitions API and CSS slide-fade fallbacks.
- **Charts**: Chart.js v4.4.1 (CDN-loaded) configured with custom luminous quantile bands ($P_{10} \text{--} P_{90}$ area fills, $P_{50}$ median lines, and historical observations).
- **Design Tokens**: Defined in `styles.css` using CSS custom properties with light and dark mode toggles:
  - Base Ink: `#0B1120` (Dark) / `#F8FAFC` (Light)
  - Primary Accent: Teal (`#0D9488` / `#2DD4BF`)
  - Warning/Reorder: Amber (`#D97706` / `#F59E0B`)
  - Critical Stockout: Red (`#DC2626` / `#F87171`)
  - Healthy State: Green (`#16A34A` / `#34D399`)
- **Typography**: Inter / system sans-serif with strict `font-variant-numeric: tabular-nums` across all numeric tables and KPI metrics to eliminate tabular jitter.

---

## 3. The Visual Layer & Design Philosophy

The visual design is disciplined, calm, and data-first:

### 1. Focused Enterprise Design System (`styles.css`)
- **Restraint Over Clutter**: Extraneous 3D meshes and non-essential background animations have been stripped away so managers can immediately focus on the numbers, uncertainty intervals, and reorder alerts.
- **Deep Slate Architecture**: Deep obsidian ink surfaces (`#0B1120` in dark mode) paired with off-white (`#F8FAFC` in light mode), razor-sharp 1px subtle borders (`#1E293B`), and high-contrast typography.
- **Data Legibility Priority**: Dense tables, chart canvas containers, and KPI cards maintain strict WCAG 2.2 AA contrast compliance.

### 2. Purposeful Interaction Design (`glass.js` & `transitions.js`)
- **Specular Border Highlights**: Interactive cards calculate real-time relative cursor positions (`--mouse-x`, `--mouse-y`) to project subtle border highlights without distracting refractions.
- **Instantaneous Load**: Zero artificial splash gates or delays; direct immediate landing onto the decision dashboard.
- **View Transitions**: Rapid, smooth route crossfades with zero layout shift, protected against unhandled promise rejections.
- **Action Feedback**: Clear toast notifications on decision acceptance/rejection, CSV validation reports, and model promotions.

---

## 4. Reference Repositories & Attribution

1. **[dashersw/liquid-glass-js](https://github.com/dashersw/liquid-glass-js)** (MIT License)
   - *Adapted*: Specular highlight cursor tracking algorithm across glass surfaces and frosted backdrop treatments.
2. **[pmndrs/react-three-fiber](https://github.com/pmndrs/react-three-fiber)** (MIT License)
   - *Adapted*: Minimal render loop lifecycle and event debouncing patterns.
   - *Adapted*: Fluid ripple distortion mechanics adapted for our custom canvas-based forecasting brand mark.

*All third-party code and algorithmic concepts are used in full compliance with their respective open-source licenses.*

---

## 5. Mandatory Scope & Domain Notices

1. **Simulation Notice**:
   > *"Stock and lead-time values are simulated/user-entered."*
   Prominently displayed across all screens displaying inventory, stock-on-hand, or lead-time parameters.
2. **Benchmark Geographic Disclaimer**:
   > The benchmark datasets (Rossmann Store Sales and Kaggle Store Item Demand) are open retail evaluation benchmarks and do not represent Pakistani retail.
3. **Product Positioning**:
   Presented exclusively as **AI Demand Forecasting and Inventory Decision Support**, never "Sales Prediction".

---

## 6. How to Run

### Direct Browser Opening
No build step, Node server, or package installation is required:
- Double-click **`index.html`** or open `file:///.../index.html` in Chrome, Edge, Firefox, or Safari.

### Local Development Server
To run over HTTP:
```bash
# Using uv (recommended)
uv run python -m http.server 8000

# Open in browser:
http://localhost:8000
```

### Accessibility & Debug Flags
- **Lite Mode**: Append `?lite=1` to the URL (`http://localhost:8000/?lite=1`) to disable WebGL 3D shaders, heavy blur, and micro-animations for ultra-low-spec hardware.
- **Reduced Motion**: Automatically honors OS-level `prefers-reduced-motion: reduce` by replacing all animations with instant state transitions.

