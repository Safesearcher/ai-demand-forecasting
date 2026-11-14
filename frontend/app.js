/**
 * AI Demand Forecasting and Inventory Decision Support
 * Application Logic & Reactive Controller
 * SRS Reference: AI_Demand_Forecasting_SRS_v3_short.md
 */

(function () {
  'use strict';

  // Global Application State
  const state = {
    currentRoute: 'dashboard',
    activeDataset: 'store-item', // 'store-item' or 'rossmann'
    selectedSeriesId: 'item-17-store-3',
    filterStatus: 'all',
    theme: localStorage.getItem('theme') || 'light',
    charts: {
      dashboard: null,
      item: null,
      horizon: null,
      calibration: null
    }
  };

  // Reorder Formula Engine (SRS Section 4)
  function computeReorder(series) {
    const L = Number(series.leadTime);
    const R = Number(series.reviewPeriod);
    const T = L + R;
    const sigmaD = Number(series.sigmaD);
    const z = Number(series.serviceLevelZ);
    const D_T = Number(series.d_T);
    const S = Number(series.stock);
    const I = Number(series.incoming);

    // SS = z * sigma_d * sqrt(T)
    const exactSS = z * sigmaD * Math.sqrt(T);
    const roundedSS = Math.round(exactSS);

    // Q = max(0, ceil(D_T + SS - S - I))
    const rawQ = D_T + exactSS - S - I;
    const Q = Math.max(0, Math.ceil(rawQ));

    // Stockout risk: S + I < P90 demand over L
    const p90L = Number(series.p90LeadTimeDemand);
    const stockoutRisk = (S + I) < p90L;

    // Status classification
    let status = 'healthy';
    let statusLabel = 'Healthy';
    if (stockoutRisk) {
      status = 'risk';
      statusLabel = 'Stockout Risk';
    } else if (Q > 0) {
      status = 'reorder';
      statusLabel = 'Reorder Recommended';
    }

    return {
      L, R, T, sigmaD, z, D_T, S, I, p90L,
      exactSS,
      roundedSS,
      rawQ,
      Q,
      stockoutRisk,
      status,
      statusLabel
    };
  }

  // Helper to get active series list
  function getActiveSeriesList() {
    return state.activeDataset === 'store-item'
      ? MOCK_DATA.storeItemSeries
      : MOCK_DATA.rossmannSeries;
  }

  function getSeriesById(id) {
    const list = getActiveSeriesList();
    return list.find(s => s.id === id) || list[0];
  }

  // Toast System
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast';
    const icon = type === 'success' ? '✓' : type === 'warning' ? '⚠' : 'ℹ';
    toast.innerHTML = `<span style="color: var(--accent); font-weight: bold;">${icon}</span> <span>${message}</span>`;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  // Dynamic Theme Colors Engine (Light Executive & Dark Preserved)
  function getThemeColors() {
    const isDark = state.theme === 'dark';
    if (isDark) {
      return {
        isDark: true,
        gridColor: '#243047',
        textColor: '#94A3B8',
        tooltipBg: '#1E293B',
        tooltipBorder: '#334155',
        tooltipTitle: '#F8FAFC',
        tooltipBody: '#F8FAFC',
        // Dashboard / Item
        histBorder: '#E2E8F0',
        histBg: '#E2E8F0',
        p50: '#FACC15',
        p10p90Border: 'rgba(250, 204, 21, 0.45)',
        bandBg: 'rgba(250, 204, 21, 0.18)',
        bandBgItem: 'rgba(250, 204, 21, 0.2)',
        // Horizon
        lgb: '#FACC15',
        sn: '#D97706',
        ma: '#64748B',
        // Calibration
        theoretical: '#334155',
        empirical: '#FACC15'
      };
    } else {
      return {
        isDark: false,
        gridColor: '#e3e6eb',
        textColor: '#5b6472',
        tooltipBg: '#ffffff',
        tooltipBorder: '#e3e6eb',
        tooltipTitle: '#14181f',
        tooltipBody: '#5b6472',
        // Dashboard / Item
        histBorder: '#64748b',
        histBg: '#64748b',
        p50: '#b7791f',
        p10p90Border: 'rgba(183, 121, 31, 0.45)',
        bandBg: 'rgba(183, 121, 31, 0.16)',
        bandBgItem: 'rgba(183, 121, 31, 0.16)',
        // Horizon
        lgb: '#b7791f',
        sn: '#2563eb',
        ma: '#94a3b8',
        // Calibration
        theoretical: '#cbd5e1',
        empirical: '#b7791f'
      };
    }
  }

  // Theme Switching & Adaptive Chart Re-rendering
  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    // Redraw charts with adapted colors
    Object.values(state.charts).forEach(chart => {
      if (chart) {
        updateChartTheme(chart);
      }
    });
  }

  function updateChartTheme(chart) {
    const colors = getThemeColors();
    if (chart.options.scales) {
      Object.values(chart.options.scales).forEach(scale => {
        if (scale.grid) scale.grid.color = colors.gridColor;
        if (scale.ticks) scale.ticks.color = colors.textColor;
        if (scale.title) scale.title.color = colors.textColor;
      });
    }
    if (chart.options.plugins && chart.options.plugins.tooltip) {
      chart.options.plugins.tooltip.backgroundColor = colors.tooltipBg;
      chart.options.plugins.tooltip.borderColor = colors.tooltipBorder;
      chart.options.plugins.tooltip.titleColor = colors.tooltipTitle;
      chart.options.plugins.tooltip.bodyColor = colors.tooltipBody;
    }
    if (chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {
      chart.options.plugins.legend.labels.color = colors.textColor;
    }
    // Update dataset colors based on chart type
    if (chart.canvas && chart.canvas.id === 'dashboard-history-chart') {
      if (chart.data.datasets[0]) {
        chart.data.datasets[0].borderColor = colors.histBorder;
        chart.data.datasets[0].backgroundColor = colors.histBg;
      }
      if (chart.data.datasets[1]) {
        chart.data.datasets[1].borderColor = colors.p50;
        chart.data.datasets[1].backgroundColor = colors.p50;
      }
      if (chart.data.datasets[2]) {
        chart.data.datasets[2].borderColor = colors.p10p90Border;
      }
      if (chart.data.datasets[3]) {
        chart.data.datasets[3].borderColor = colors.p10p90Border;
        chart.data.datasets[3].backgroundColor = colors.bandBg;
      }
    } else if (chart.canvas && chart.canvas.id === 'item-uncertainty-chart') {
      if (chart.data.datasets[0]) {
        chart.data.datasets[0].borderColor = colors.p50;
        chart.data.datasets[0].backgroundColor = colors.p50;
      }
      if (chart.data.datasets[1]) {
        chart.data.datasets[1].borderColor = colors.p10p90Border;
      }
      if (chart.data.datasets[2]) {
        chart.data.datasets[2].borderColor = colors.p10p90Border;
        chart.data.datasets[2].backgroundColor = colors.bandBgItem;
      }
    } else if (chart.canvas && chart.canvas.id === 'horizon-error-chart') {
      if (chart.data.datasets[0]) {
        chart.data.datasets[0].borderColor = colors.lgb;
        chart.data.datasets[0].backgroundColor = colors.lgb;
      }
      if (chart.data.datasets[1]) {
        chart.data.datasets[1].borderColor = colors.sn;
        chart.data.datasets[1].backgroundColor = colors.sn;
      }
      if (chart.data.datasets[2]) {
        chart.data.datasets[2].borderColor = colors.ma;
        chart.data.datasets[2].backgroundColor = colors.ma;
      }
    } else if (chart.canvas && chart.canvas.id === 'calibration-chart') {
      if (chart.data.datasets[0]) {
        chart.data.datasets[0].backgroundColor = colors.theoretical;
      }
      if (chart.data.datasets[1]) {
        chart.data.datasets[1].backgroundColor = colors.empirical;
      }
    }
    chart.update();
  }

  // Router Engine with View Transitions Support
  function handleRoute() {
    const hash = window.location.hash || '#/dashboard';
    const [pathPart, queryPart] = hash.split('?');
    const route = pathPart.replace('#/', '') || 'dashboard';

    // Parse query params if any
    if (queryPart) {
      const params = new URLSearchParams(queryPart);
      if (params.has('id')) {
        state.selectedSeriesId = params.get('id');
      }
    }

    state.currentRoute = route;

    const updateViews = () => {
      // Update active nav links
      document.querySelectorAll('.nav-link').forEach(link => {
        const linkRoute = link.getAttribute('href').replace('#/', '');
        link.classList.toggle('active', linkRoute === route);
      });

      // Update views visibility
      document.querySelectorAll('.page-view').forEach(view => {
        view.classList.remove('active');
      });

      const activeView = document.getElementById(`view-${route}`);
      if (activeView) {
        activeView.classList.add('active');
      } else {
        document.getElementById('view-dashboard').classList.add('active');
        state.currentRoute = 'dashboard';
      }

      // Render route specific content
      renderCurrentRoute();
    };

    if (window.appMotion && window.appMotion.performViewTransition) {
      window.appMotion.performViewTransition(updateViews);
    } else {
      updateViews();
    }

    // Scroll to top smoothly
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function renderCurrentRoute() {
    switch (state.currentRoute) {
      case 'dashboard':
        renderDashboard();
        break;
      case 'item':
        renderItemDetail();
        break;
      case 'evaluation':
        renderEvaluation();
        break;
      case 'upload':
        renderUpload();
        break;
      case 'models':
        renderModels();
        break;
    }
  }

  // =========================================================================
  // DASHBOARD RENDERER (FR-09, FR-10)
  // =========================================================================
  function renderDashboard() {
    const seriesList = getActiveSeriesList();
    let attentionCount = 0;
    let unitsToOrder = 0;
    let stockoutCount = 0;

    const computedList = seriesList.map(s => {
      const calc = computeReorder(s);
      if (calc.Q > 0) attentionCount++;
      unitsToOrder += calc.Q;
      if (calc.stockoutRisk) stockoutCount++;
      return { series: s, calc };
    });

    // Sort by priority: Stockout Risk first, then largest Q, then healthy
    computedList.sort((a, b) => {
      if (a.calc.stockoutRisk && !b.calc.stockoutRisk) return -1;
      if (!a.calc.stockoutRisk && b.calc.stockoutRisk) return 1;
      return b.calc.Q - a.calc.Q;
    });

    // Update KPIs
    const kpiAttention = document.getElementById('kpi-attention-count');
    const kpiUnits = document.getElementById('kpi-units-to-order');
    const kpiStockout = document.getElementById('kpi-stockout-count');
    if (kpiAttention) kpiAttention.textContent = attentionCount;
    if (kpiUnits) kpiUnits.textContent = unitsToOrder.toLocaleString();
    if (kpiStockout) kpiStockout.textContent = stockoutCount;

    // Trigger KPI count-up animation
    if (window.appMotion && window.appMotion.triggerKpiCountUps) {
      window.appMotion.triggerKpiCountUps();
    }

    // Active Model Badge
    const activeModel = MOCK_DATA.models.find(m => m.isCurrent) || MOCK_DATA.models[0];
    const modelBadge = document.getElementById('dashboard-active-model-badge');
    if (modelBadge) {
      modelBadge.textContent = `Active Model: ${activeModel.version} LightGBM`;
    }

    // Stockout Alerts Container
    const alertsContainer = document.getElementById('dashboard-alerts-container');
    if (alertsContainer) {
      const riskItems = computedList.filter(item => item.calc.stockoutRisk);
      if (riskItems.length === 0) {
        alertsContainer.innerHTML = '';
      } else {
        alertsContainer.innerHTML = riskItems.map(item => `
          <div class="alert-row">
            <div class="alert-content">
              <span class="alert-icon" aria-hidden="true">&#9888;</span>
              <div class="alert-text">
                <strong>CRITICAL STOCKOUT RISK:</strong> ${item.series.name} (Store ${item.series.store}) has only ${item.series.stock} units in stock with ${item.series.incoming} incoming, failing to cover P90 demand over ${item.series.leadTime}-day lead time (${item.calc.p90L} units).
              </div>
            </div>
            <a href="#/item?id=${item.series.id}" class="btn btn-danger" style="font-size: 0.75rem; padding: 0.25rem 0.6rem; white-space: nowrap;">
              Review Order &rarr;
            </a>
          </div>
        `).join('');
      }
    }

    // Recommendations Table
    const tbody = document.getElementById('reorder-table-body');
    if (tbody) {
      const filtered = computedList.filter(item => {
        if (state.filterStatus === 'all') return true;
        if (state.filterStatus === 'risk') return item.calc.stockoutRisk;
        if (state.filterStatus === 'reorder') return item.calc.Q > 0 && !item.calc.stockoutRisk;
        if (state.filterStatus === 'healthy') return item.calc.Q === 0;
        return true;
      });

      if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-muted); padding: 2rem;">No series match the selected filter.</td></tr>`;
      } else {
        tbody.innerHTML = filtered.map(item => {
          const s = item.series;
          const c = item.calc;
          const badgeClass = c.stockoutRisk ? 'badge-risk' : (c.Q > 0 ? 'badge-reorder' : 'badge-healthy');
          const orderStyle = c.Q > 0 ? 'color: var(--amber); font-weight: 700;' : 'color: var(--text-muted);';

          return `
            <tr onclick="window.location.hash='#/item?id=${s.id}'">
              <td><strong>${s.id}</strong></td>
              <td>
                <div style="font-weight: 600;">${s.name}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">Store ${s.store} &bull; ${s.category || 'Retail Assortment'}</div>
              </td>
              <td class="align-right tabular-nums">${s.stock}</td>
              <td class="align-right tabular-nums">${s.incoming}</td>
              <td class="align-right tabular-nums">${s.leadTime} / ${s.reviewPeriod} d</td>
              <td class="align-right tabular-nums">${c.D_T}</td>
              <td class="align-right tabular-nums">${c.roundedSS}</td>
              <td class="align-right tabular-nums" style="${orderStyle}">${c.Q}</td>
              <td><span class="badge ${badgeClass}">${c.statusLabel}</span></td>
              <td class="align-right">
                <a href="#/item?id=${s.id}" class="item-link" onclick="event.stopPropagation();">
                  Decide &rarr;
                </a>
              </td>
            </tr>
          `;
        }).join('');
      }
    }

    // Chart.js History vs Forecast Chart
    renderDashboardChart();
  }

  function renderDashboardChart() {
    const canvas = document.getElementById('dashboard-history-chart');
    if (!canvas) return;

    // Use Item 17 @ Store 3 data as benchmark
    const series = getSeriesById('item-17-store-3');
    const titleEl = document.getElementById('chart-focused-series-name');
    if (titleEl) titleEl.textContent = `${series.name} (Store ${series.store})`;

    const history = series.history || [];
    const forecast = series.forecast || [];

    // Slice last 45 days of history for clear visual rhythm + 28 days forecast
    const recentHistory = history.slice(-45);
    const historyLabels = recentHistory.map(d => d.date);
    const forecastLabels = forecast.map(d => d.date);
    const allLabels = [...historyLabels, ...forecastLabels];

    const historyPoints = [...recentHistory.map(d => d.actual), ...forecast.map(() => null)];
    // Forecast connects from last observed point
    const lastHistVal = recentHistory[recentHistory.length - 1]?.actual || 12;
    const p50Points = [...recentHistory.map((_, i) => i === recentHistory.length - 1 ? lastHistVal : null), ...forecast.map(d => d.p50)];
    const p10Points = [...recentHistory.map((_, i) => i === recentHistory.length - 1 ? lastHistVal : null), ...forecast.map(d => d.p10)];
    const p90Points = [...recentHistory.map((_, i) => i === recentHistory.length - 1 ? lastHistVal : null), ...forecast.map(d => d.p90)];

    if (state.charts.dashboard) {
      state.charts.dashboard.destroy();
    }

    const colors = getThemeColors();

    const ctx = canvas.getContext('2d');
    state.charts.dashboard = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'Observed Daily Sales',
            data: historyPoints,
            borderColor: colors.histBorder,
            backgroundColor: colors.histBg,
            borderWidth: 2,
            pointRadius: 2,
            pointHoverRadius: 4,
            tension: 0.1
          },
          {
            label: 'P50 Forecast (Median)',
            data: p50Points,
            borderColor: colors.p50,
            backgroundColor: colors.p50,
            borderWidth: 2.5,
            borderDash: [4, 4],
            pointRadius: 3,
            tension: 0.2
          },
          {
            label: 'P10 Lower Bound',
            data: p10Points,
            borderColor: colors.p10p90Border,
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            tension: 0.2
          },
          {
            label: 'P90 Upper Bound (Uncertainty Band)',
            data: p90Points,
            borderColor: colors.p10p90Border,
            borderWidth: 1,
            pointRadius: 0,
            fill: '-1', // Fill down to P10
            backgroundColor: colors.bandBg,
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            titleColor: colors.tooltipTitle,
            bodyColor: colors.tooltipBody,
            padding: 10,
            boxPadding: 4,
            usePointStyle: true,
            callbacks: {
              label: function (ctx) {
                if (ctx.parsed.y === null) return '';
                return ` ${ctx.dataset.label}: ${ctx.parsed.y} units`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: colors.gridColor },
            ticks: {
              color: colors.textColor,
              maxTicksLimit: 12
            }
          },
          y: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.textColor },
            title: {
              display: true,
              text: 'Demand / Sales (Units)',
              color: colors.textColor
            }
          }
        }
      }
    });
  }

  // =========================================================================
  // ITEM DETAIL CONTROLLER (FR-07, FR-08, FR-12)
  // =========================================================================
  function renderItemDetail() {
    const seriesList = getActiveSeriesList();
    const select = document.getElementById('detail-series-select');

    if (select) {
      select.innerHTML = seriesList.map(s => `
        <option value="${s.id}" ${s.id === state.selectedSeriesId ? 'selected' : ''}>
          ${s.name} (Store ${s.store})
        </option>
      `).join('');

      select.onchange = (e) => {
        state.selectedSeriesId = e.target.value;
        populateItemInputs();
      };
    }

    populateItemInputs();
    renderDecisionLog();
  }

  function populateItemInputs() {
    const series = getSeriesById(state.selectedSeriesId);
    if (!series) return;

    // Header updates
    const titleEl = document.getElementById('item-display-title');
    if (titleEl) {
      titleEl.textContent = `Item ${series.item} @ Store ${series.store} \u2022 ${series.name}`;
    }

    // Set Form Inputs
    const inL = document.getElementById('input-lead-time');
    const inR = document.getElementById('input-review-period');
    const inS = document.getElementById('input-stock-on-hand');
    const inI = document.getElementById('input-incoming-orders');
    const inZ = document.getElementById('input-service-level-z');
    const inSigma = document.getElementById('input-sigma-d');
    const inDT = document.getElementById('input-demand-forecast');

    if (inL) inL.value = series.leadTime;
    if (inR) inR.value = series.reviewPeriod;
    if (inS) inS.value = series.stock;
    if (inI) inI.value = series.incoming;
    if (inZ) inZ.value = series.serviceLevelZ;
    if (inSigma) inSigma.value = series.sigmaD;
    if (inDT) inDT.value = series.d_T;

    // Attach dynamic input listeners
    [inL, inR, inS, inI, inZ, inSigma, inDT].forEach(input => {
      if (input) {
        input.oninput = recalculateLiveItemFormula;
      }
    });

    recalculateLiveItemFormula();
    renderItemUncertaintyChart(series);
  }

  function recalculateLiveItemFormula() {
    const inL = document.getElementById('input-lead-time');
    const inR = document.getElementById('input-review-period');
    const inS = document.getElementById('input-stock-on-hand');
    const inI = document.getElementById('input-incoming-orders');
    const inZ = document.getElementById('input-service-level-z');
    const inSigma = document.getElementById('input-sigma-d');
    const inDT = document.getElementById('input-demand-forecast');

    const L = Number(inL.value) || 1;
    const R = Number(inR.value) || 1;
    const S = Number(inS.value) || 0;
    const I = Number(inI.value) || 0;
    const z = Number(inZ.value) || 1.65;
    const sigmaD = Number(inSigma.value) || 1;
    const D_T = Number(inDT.value) || 0;

    const series = getSeriesById(state.selectedSeriesId);
    const p90L = series ? series.p90LeadTimeDemand : 48;

    // Calculations
    const T = L + R;
    const sqrtT = Math.sqrt(T);
    const exactSS = z * sigmaD * sqrtT;
    const roundedSS = Math.round(exactSS);
    const rawQ = D_T + exactSS - S - I;
    const Q = Math.max(0, Math.ceil(rawQ));
    const stockoutRisk = (S + I) < p90L;

    // Update metadata banner
    const metaStock = document.getElementById('meta-stock');
    const metaIncoming = document.getElementById('meta-incoming');
    const metaLR = document.getElementById('meta-lr');
    const metaDT = document.getElementById('meta-dt');
    if (metaStock) metaStock.textContent = S;
    if (metaIncoming) metaIncoming.textContent = I;
    if (metaLR) metaLR.textContent = `${L} / ${R} days`;
    if (metaDT) metaDT.textContent = `${D_T} units`;

    // Update formula breakdown step text
    document.getElementById('step-t-val').textContent = `${L} + ${R} = ${T} days`;
    document.getElementById('step-ss-val').textContent = `${z} \u00d7 ${sigmaD} \u00d7 ${sqrtT.toFixed(2)} \u2248 ${exactSS.toFixed(2)} \u2248 ${roundedSS}`;
    document.getElementById('step-net-val').textContent = `${D_T} + ${roundedSS} = ${D_T + roundedSS}`;
    document.getElementById('step-pos-val').textContent = `${S} + ${I} = ${S + I}`;
    document.getElementById('step-q-val').textContent = `${Q} Units`;

    // Decision Callout Box updates
    const calloutBox = document.getElementById('decision-callout-box');
    const calloutStatus = document.getElementById('callout-status-label');
    const calloutBadge = document.getElementById('callout-risk-badge');
    const liveOrderQ = document.getElementById('live-order-q');
    const headerBadge = document.getElementById('item-header-badge');
    const acceptBtnQty = document.getElementById('btn-accept-qty');
    const expl = document.getElementById('live-recommendation-explanation');

    if (liveOrderQ) liveOrderQ.textContent = `${Q} UNITS`;
    if (acceptBtnQty) acceptBtnQty.textContent = Q;

    if (stockoutRisk) {
      if (calloutBox) calloutBox.classList.add('risk-active');
      if (calloutStatus) calloutStatus.textContent = '\u2022 Critical Stockout Risk Alert';
      if (calloutBadge) {
        calloutBadge.style.display = 'inline-flex';
        calloutBadge.textContent = 'Stockout Risk Flagged';
      }
      if (headerBadge) {
        headerBadge.className = 'badge badge-risk';
        headerBadge.textContent = 'Stockout Risk';
      }
      if (expl) {
        expl.innerHTML = `Position (S + I = <strong>${S + I}</strong>) is insufficient for P90 lead-time demand (<strong>${p90L}</strong>). Reorder <strong>${Q} units</strong> immediately.`;
      }
    } else if (Q > 0) {
      if (calloutBox) calloutBox.classList.remove('risk-active');
      if (calloutStatus) calloutStatus.textContent = '\u2022 Reorder Recommended';
      if (calloutBadge) calloutBadge.style.display = 'none';
      if (headerBadge) {
        headerBadge.className = 'badge badge-reorder';
        headerBadge.textContent = 'Reorder Recommended';
      }
      if (expl) {
        expl.innerHTML = `Lead-time position is stable, but net position falls below target safety buffer (${roundedSS} units). Place order for <strong>${Q} units</strong>.`;
      }
    } else {
      if (calloutBox) calloutBox.classList.remove('risk-active');
      if (calloutStatus) calloutStatus.textContent = '\u2022 Inventory Position Healthy';
      if (calloutBadge) calloutBadge.style.display = 'none';
      if (headerBadge) {
        headerBadge.className = 'badge badge-healthy';
        headerBadge.textContent = 'Healthy (Q = 0)';
      }
      if (expl) {
        expl.innerHTML = `Existing inventory position (<strong>${S + I}</strong>) fully covers 10-day demand (${D_T}) and safety stock (${roundedSS}). No order needed.`;
      }
    }

    // Bind Action Buttons
    bindDecisionActions(Q);
  }

  function bindDecisionActions(currentQ) {
    const series = getSeriesById(state.selectedSeriesId);
    const seriesName = series ? `${series.name} (Store ${series.store})` : state.selectedSeriesId;

    // Reset button
    const btnReset = document.getElementById('btn-reset-worked-example');
    if (btnReset) {
      btnReset.onclick = () => {
        document.getElementById('input-lead-time').value = 3;
        document.getElementById('input-review-period').value = 7;
        document.getElementById('input-stock-on-hand').value = 40;
        document.getElementById('input-incoming-orders').value = 0;
        document.getElementById('input-service-level-z').value = 1.65;
        document.getElementById('input-sigma-d').value = 5.7;
        document.getElementById('input-demand-forecast').value = 120;
        recalculateLiveItemFormula();
        showToast('Reset to SRS worked example: L=3, R=7, D_T=120, z=1.65, S=40 -> Q=110', 'info');
      };
    }

    // Accept Action
    const btnAccept = document.getElementById('btn-accept-rec');
    if (btnAccept) {
      btnAccept.onclick = () => {
        addDecisionRecord({
          series: seriesName,
          action: 'Accept',
          recommendedQ: currentQ,
          finalQ: currentQ,
          reason: 'Accepted system recommended quantity based on standard 95% safety stock.'
        });
        showToast(`Order accepted: ${currentQ} units for ${seriesName}`, 'success');
      };
    }

    // Adjust Action Modal
    const btnAdjust = document.getElementById('btn-adjust-rec');
    const modal = document.getElementById('modal-adjust');
    const btnCloseAdjust = document.getElementById('btn-close-adjust');
    const btnCancelAdjust = document.getElementById('btn-cancel-adjust');
    const btnSaveAdjust = document.getElementById('btn-save-adjust');
    const modalRecQty = document.getElementById('modal-adjust-rec-qty');
    const modalInputQty = document.getElementById('modal-input-adjusted-qty');
    const modalInputReason = document.getElementById('modal-input-adjust-reason');

    if (btnAdjust && modal) {
      btnAdjust.onclick = () => {
        modalRecQty.textContent = `${currentQ} units`;
        modalInputQty.value = currentQ;
        modalInputReason.value = '';
        modal.classList.add('open');
      };

      const closeModal = () => modal.classList.remove('open');
      if (btnCloseAdjust) btnCloseAdjust.onclick = closeModal;
      if (btnCancelAdjust) btnCancelAdjust.onclick = closeModal;

      if (btnSaveAdjust) {
        btnSaveAdjust.onclick = () => {
          const adjVal = Number(modalInputQty.value);
          const reason = modalInputReason.value.trim() || 'Manager manual operational adjustment';
          addDecisionRecord({
            series: seriesName,
            action: 'Adjust',
            recommendedQ: currentQ,
            finalQ: adjVal,
            reason: reason
          });
          closeModal();
          showToast(`Order adjusted to ${adjVal} units for ${seriesName}`, 'info');
        };
      }
    }

    // Reject Action
    const btnReject = document.getElementById('btn-reject-rec');
    if (btnReject) {
      btnReject.onclick = () => {
        const reason = prompt('Please enter justification for rejecting recommendation:', 'Sufficient stock buffer / batch delivery rescheduled');
        if (reason !== null) {
          addDecisionRecord({
            series: seriesName,
            action: 'Reject',
            recommendedQ: currentQ,
            finalQ: 0,
            reason: reason || 'Manager rejected purchase order.'
          });
          showToast(`Recommendation rejected for ${seriesName}`, 'warning');
        }
      };
    }
  }

  function addDecisionRecord(rec) {
    const now = new Date();
    const pad = n => String(n).padStart(2, '0');
    const timeStr = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}`;

    MOCK_DATA.decisions.unshift({
      id: `dec-${Date.now().toString().slice(-4)}`,
      timestamp: timeStr,
      series: rec.series,
      action: rec.action,
      recommendedQ: rec.recommendedQ,
      finalQ: rec.finalQ,
      reason: rec.reason,
      user: 'Manager (Session)'
    });

    renderDecisionLog();
  }

  function renderDecisionLog() {
    const tbody = document.getElementById('decision-log-body');
    if (!tbody) return;

    if (MOCK_DATA.decisions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No decisions recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = MOCK_DATA.decisions.map(d => {
      let badgeClass = 'badge-model';
      if (d.action === 'Accept') badgeClass = 'badge-healthy';
      if (d.action === 'Adjust') badgeClass = 'badge-reorder';
      if (d.action === 'Reject') badgeClass = 'badge-risk';

      return `
        <tr>
          <td class="tabular-nums" style="font-size: 0.8rem; color: var(--text-muted);">${d.timestamp}</td>
          <td><strong>${d.series}</strong></td>
          <td><span class="badge ${badgeClass}">${d.action}</span></td>
          <td class="align-right tabular-nums">${d.recommendedQ}</td>
          <td class="align-right tabular-nums" style="font-weight: 700;">${d.finalQ}</td>
          <td style="font-size: 0.8rem; max-width: 280px;">${d.reason}</td>
          <td style="font-size: 0.8rem; color: var(--text-muted);">${d.user}</td>
        </tr>
      `;
    }).join('');
  }

  function renderItemUncertaintyChart(series) {
    const canvas = document.getElementById('item-uncertainty-chart');
    if (!canvas) return;

    if (state.charts.item) {
      state.charts.item.destroy();
    }

    const colors = getThemeColors();

    // 10-day protection window breakdown
    const labels = Array.from({ length: 10 }, (_, i) => `Day +${i + 1}`);
    const forecast = series.forecast ? series.forecast.slice(0, 10) : [];
    const p10 = forecast.map(f => f.p10);
    const p50 = forecast.map(f => f.p50);
    const p90 = forecast.map(f => f.p90);

    const ctx = canvas.getContext('2d');
    state.charts.item = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'P50 (Median)',
            data: p50,
            borderColor: colors.p50,
            backgroundColor: colors.p50,
            borderWidth: 2,
            pointRadius: 3
          },
          {
            label: 'P10 Lower Bound',
            data: p10,
            borderColor: colors.p10p90Border,
            borderWidth: 1,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'P90 Upper Bound',
            data: p90,
            borderColor: colors.p10p90Border,
            borderWidth: 1,
            pointRadius: 0,
            fill: '-1',
            backgroundColor: colors.bandBgItem
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            titleColor: colors.tooltipTitle,
            bodyColor: colors.tooltipBody,
            padding: 8
          }
        },
        scales: {
          x: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.textColor }
          },
          y: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.textColor },
            title: { display: true, text: 'Daily Units', color: colors.textColor }
          }
        }
      }
    });
  }

  // =========================================================================
  // EVALUATION VIEW CONTROLLER (FR-11)
  // =========================================================================
  function renderEvaluation() {
    const compTable = document.getElementById('evaluation-table-body');
    if (compTable) {
      compTable.innerHTML = MOCK_DATA.evaluation.modelComparison.map(m => `
        <tr>
          <td>
            <div style="font-weight: 700; color: var(--text-main);">${m.model}</div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">Training: ${m.trainingTime}</div>
          </td>
          <td><span class="badge ${m.model.includes('LightGBM') ? 'badge-healthy' : 'badge-model'}">${m.type}</span></td>
          <td class="align-right tabular-nums">${m.mae.toFixed(2)}</td>
          <td class="align-right tabular-nums" style="font-weight: 700;">${m.rmse.toFixed(2)}</td>
          <td class="align-right tabular-nums">${m.wmape.toFixed(1)}%</td>
          <td class="align-right tabular-nums">${m.mase.toFixed(2)}</td>
          <td class="align-right tabular-nums" style="color: var(--accent); font-weight: 700;">${m.coverageP10P90.toFixed(1)}%</td>
          <td class="tabular-nums" style="font-size: 0.8rem; color: var(--text-muted);">${m.inferenceLatency}</td>
        </tr>
      `).join('');
    }

    renderHorizonErrorChart();
    renderCalibrationChart();
  }

  function renderHorizonErrorChart() {
    const canvas = document.getElementById('horizon-error-chart');
    if (!canvas) return;

    if (state.charts.horizon) {
      state.charts.horizon.destroy();
    }

    const colors = getThemeColors();

    const horizons = MOCK_DATA.evaluation.horizonError;
    const labels = horizons.map(h => h.day);
    const lgbMAE = horizons.map(h => h.lgbMAE);
    const snMAE = horizons.map(h => h.snMAE);
    const maMAE = horizons.map(h => h.maMAE);

    const ctx = canvas.getContext('2d');
    state.charts.horizon = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'LightGBM Quantile (Ours)',
            data: lgbMAE,
            borderColor: colors.lgb,
            backgroundColor: colors.lgb,
            borderWidth: 2.5,
            pointRadius: 3
          },
          {
            label: 'Seasonal Naive (Lag 7)',
            data: snMAE,
            borderColor: colors.sn,
            backgroundColor: colors.sn,
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 2
          },
          {
            label: '14-Day Moving Average',
            data: maMAE,
            borderColor: colors.ma,
            backgroundColor: colors.ma,
            borderWidth: 1.5,
            borderDash: [2, 2],
            pointRadius: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: { color: colors.textColor, boxWidth: 12, padding: 12 }
          },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            titleColor: colors.tooltipTitle,
            bodyColor: colors.tooltipBody,
            padding: 8
          }
        },
        scales: {
          x: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.textColor }
          },
          y: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.textColor },
            title: { display: true, text: 'Holdout MAE', color: colors.textColor }
          }
        }
      }
    });
  }

  function renderCalibrationChart() {
    const canvas = document.getElementById('calibration-chart');
    if (!canvas) return;

    if (state.charts.calibration) {
      state.charts.calibration.destroy();
    }

    const colors = getThemeColors();

    const calData = MOCK_DATA.evaluation.calibration;
    const labels = calData.map(c => c.quantile);
    const nominal = calData.map(c => c.nominal);
    const empirical = calData.map(c => c.empirical);

    const ctx = canvas.getContext('2d');
    state.charts.calibration = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Theoretical Target (%)',
            data: nominal,
            backgroundColor: colors.theoretical,
            borderRadius: 4
          },
          {
            label: 'Empirical Coverage (%)',
            data: empirical,
            backgroundColor: colors.empirical,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: { color: colors.textColor, boxWidth: 12, padding: 12 }
          },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            titleColor: colors.tooltipTitle,
            bodyColor: colors.tooltipBody,
            padding: 8
          }
        },
        scales: {
          x: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.textColor }
          },
          y: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.textColor },
            max: 100,
            title: { display: true, text: 'Interval Coverage %', color: colors.textColor }
          }
        }
      }
    });
  }

  // =========================================================================
  // DATA UPLOAD & CLIENT-SIDE VALIDATION (FR-01, FR-02)
  // =========================================================================
  function renderUpload() {
    const schemaName = document.getElementById('upload-schema-name');
    if (schemaName) {
      schemaName.textContent = state.activeDataset === 'store-item'
        ? 'Store Item Demand (date, store, item, sales)'
        : 'Rossmann Store Sales (Date, Store, DayOfWeek, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday)';
    }

    const dropzone = document.getElementById('csv-dropzone');
    const fileInput = document.getElementById('csv-file-input');

    if (dropzone && fileInput) {
      dropzone.ondragover = (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      };
      dropzone.ondragleave = () => {
        dropzone.classList.remove('dragover');
      };
      dropzone.ondrop = (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
          handleCsvFile(e.dataTransfer.files[0]);
        }
      };

      fileInput.onchange = (e) => {
        if (e.target.files.length > 0) {
          handleCsvFile(e.target.files[0]);
        }
      };
    }

    // Sample Load Buttons
    const btnSampleValid = document.getElementById('btn-load-sample-valid');
    const btnSampleInvalid = document.getElementById('btn-load-sample-invalid');

    if (btnSampleValid) {
      btnSampleValid.onclick = () => {
        const validSample = generateValidCsvSample(state.activeDataset);
        validateCsvText(validSample, 'sample-valid.csv');
      };
    }

    if (btnSampleInvalid) {
      btnSampleInvalid.onclick = () => {
        const invalidSample = generateInvalidCsvSample();
        validateCsvText(invalidSample, 'sample-corrupted-errors.csv');
      };
    }
  }

  function handleCsvFile(file) {
    if (!file.name.endsWith('.csv')) {
      showToast('Please upload a valid .csv file.', 'warning');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      validateCsvText(e.target.result, file.name);
    };
    reader.readAsText(file);
  }

  function generateValidCsvSample(dataset) {
    if (dataset === 'store-item') {
      return `date,store,item,sales
2026-10-01,1,1,15
2026-10-01,1,2,22
2026-10-01,3,17,14
2026-10-02,1,1,18
2026-10-02,1,2,19
2026-10-02,3,17,12
2026-10-03,1,1,25
2026-10-03,1,2,30
2026-10-03,3,17,16`;
    } else {
      return `Date,Store,DayOfWeek,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday
2026-10-01,1,4,5200,560,1,1,0,0
2026-10-02,1,5,5600,610,1,1,0,0
2026-10-03,1,6,6100,680,1,0,0,0
2026-10-04,1,7,0,0,0,0,0,0
2026-10-05,1,1,4900,530,1,0,0,0`;
    }
  }

  function generateInvalidCsvSample() {
    return `timestamp,store_id,item_code,revenue
2026-13-45,Store_One,NaN,-50.2
2026-10-01,1,1,15
2026-10-01,1,1,15
invalid-date,1,2,twenty-two`;
  }

  function validateCsvText(text, fileName) {
    const reportArea = document.getElementById('validation-report-area');
    if (!reportArea) return;

    const lines = text.trim().split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length < 2) {
      reportArea.innerHTML = `
        <div class="validation-card invalid">
          <div class="validation-header">
            <span style="color: var(--red); font-size: 1.25rem;">&#10006;</span>
            <span class="validation-title">CSV Validation Failed: Empty or Missing Header</span>
          </div>
          <p style="font-size: 0.85rem; color: var(--red-text);">File contains fewer than 2 lines.</p>
        </div>
      `;
      return;
    }

    const header = lines[0].split(',').map(h => h.trim());
    const expectedCols = state.activeDataset === 'store-item'
      ? ['date', 'store', 'item', 'sales']
      : ['date', 'store', 'dayofweek', 'sales', 'customers', 'open', 'promo', 'stateholiday', 'schoolholiday'];

    const errors = [];
    const lowerHeader = header.map(h => h.toLowerCase());

    // Check missing columns
    expectedCols.forEach(col => {
      if (!lowerHeader.includes(col)) {
        errors.push(`Missing mandatory column: <code>${col}</code>`);
      }
    });

    const seenKeys = new Set();
    let duplicateCount = 0;
    let invalidDates = 0;
    let invalidNumbers = 0;

    // Check rows if schema matches reasonably
    if (errors.length === 0) {
      for (let i = 1; i < lines.length; i++) {
        const row = lines[i].split(',').map(r => r.trim());
        if (row.length !== header.length) {
          errors.push(`Line ${i + 1}: Column count mismatch (expected ${header.length}, got ${row.length})`);
          if (errors.length > 6) break;
          continue;
        }

        const dateVal = row[0];
        // Validate date format YYYY-MM-DD
        const isDateValid = /^\d{4}-\d{2}-\d{2}$/.test(dateVal) && !isNaN(Date.parse(dateVal));
        if (!isDateValid) invalidDates++;

        // Duplicate check (Key = date + store + item)
        const key = `${dateVal}_${row[1]}_${row[2] || ''}`;
        if (seenKeys.has(key)) {
          duplicateCount++;
        } else {
          seenKeys.add(key);
        }

        // Numeric checks for sales
        const salesVal = Number(row[state.activeDataset === 'store-item' ? 3 : 3]);
        if (isNaN(salesVal) || salesVal < 0) {
          invalidNumbers++;
        }
      }
    }

    if (invalidDates > 0) errors.push(`Detected <strong>${invalidDates}</strong> rows with invalid dates (must be YYYY-MM-DD).`);
    if (duplicateCount > 0) errors.push(`Detected <strong>${duplicateCount}</strong> duplicate records (same series and date).`);
    if (invalidNumbers > 0) errors.push(`Detected <strong>${invalidNumbers}</strong> non-numeric or negative sales values.`);

    const isValid = errors.length === 0;

    if (isValid) {
      reportArea.innerHTML = `
        <div class="validation-card valid">
          <div class="validation-header">
            <span style="color: var(--green); font-size: 1.25rem;">&#10004;</span>
            <span class="validation-title">CSV Validation Passed: ${fileName}</span>
          </div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.75rem;">
            Successfully verified ${lines.length - 1} records against <strong>${state.activeDataset === 'store-item' ? 'Store Item Demand' : 'Rossmann'}</strong> schema.
          </p>
          <ul class="validation-list">
            <li>&#10004; All required columns present (${header.join(', ')})</li>
            <li>&#10004; No invalid dates detected</li>
            <li>&#10004; Zero duplicate timestamps per series</li>
            <li>&#10004; Numeric values within expected bounds</li>
          </ul>
          <div style="margin-top: 1rem;">
            <button type="button" class="btn btn-primary" onclick="window.showToast('Data ingestion completed. SQLite local cache updated.', 'success')">
              Import &amp; Regenerate Forecasts
            </button>
          </div>
        </div>
      `;
      showToast(`CSV validated successfully: ${fileName}`, 'success');
    } else {
      reportArea.innerHTML = `
        <div class="validation-card invalid">
          <div class="validation-header">
            <span style="color: var(--red); font-size: 1.25rem;">&#10006;</span>
            <span class="validation-title">Validation Failed: ${fileName}</span>
          </div>
          <p style="font-size: 0.85rem; color: var(--red-text); margin-bottom: 0.75rem;">
            File rejected per SRS FR-02 ingestion constraints. Please fix the following ${errors.length} issue(s):
          </p>
          <ul class="validation-list">
            ${errors.map(err => `<li>&#10006; ${err}</li>`).join('')}
          </ul>
        </div>
      `;
      showToast(`CSV validation failed with ${errors.length} issue(s)`, 'warning');
    }
  }

  // =========================================================================
  // MODELS REGISTRY CONTROLLER (FR-13)
  // =========================================================================
  function renderModels() {
    const tbody = document.getElementById('models-table-body');
    if (!tbody) return;

    tbody.innerHTML = MOCK_DATA.models.map(m => {
      let statusBadge = '<span class="badge badge-model">Archived</span>';
      if (m.isCurrent) statusBadge = '<span class="badge badge-healthy">Active Model</span>';
      else if (m.status === 'validated') statusBadge = '<span class="badge badge-reorder">Validated Candidate</span>';
      else if (m.status === 'rejected') statusBadge = '<span class="badge badge-risk">Validation Rejected</span>';
      else if (m.status === 'benchmark') statusBadge = '<span class="badge badge-model">Baseline Benchmark</span>';

      let actionHtml = '';
      if (m.isCurrent) {
        actionHtml = '<span style="font-size: 0.8rem; color: var(--green); font-weight: 600;">&#10004; Active in Production</span>';
      } else if (m.canSwitch) {
        actionHtml = `
          <button type="button" class="btn btn-primary" style="font-size: 0.75rem; padding: 0.25rem 0.6rem;" onclick="window.switchActiveModel('${m.id}')">
            Switch to Active
          </button>
        `;
      } else {
        const reason = m.status === 'rejected' ? 'Blocked: RMSE degraded compared to current version' : 'Baseline / Archived';
        actionHtml = `
          <button type="button" class="btn btn-outline" style="font-size: 0.75rem; padding: 0.25rem 0.6rem;" disabled title="${reason}">
            Switch Disabled
          </button>
        `;
      }

      return `
        <tr style="${m.isCurrent ? 'background-color: var(--accent-subtle);' : ''}">
          <td>
            <strong>${m.version}</strong>
            ${m.isCurrent ? '<span style="color: var(--accent); margin-left: 0.25rem;">&#9733;</span>' : ''}
          </td>
          <td>
            <div style="font-weight: 600;">${m.name}</div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">${m.notes}</div>
          </td>
          <td class="tabular-nums" style="font-size: 0.8rem;">${m.trainedDate}</td>
          <td style="font-size: 0.75rem; color: var(--text-muted);">${m.validationWindow}</td>
          <td class="align-right tabular-nums" style="font-weight: 700;">${m.rmse.toFixed(2)}</td>
          <td class="align-right tabular-nums">${m.mae.toFixed(2)}</td>
          <td class="align-right tabular-nums" style="color: var(--accent); font-weight: 600;">${m.coverage.toFixed(1)}%</td>
          <td>${statusBadge}</td>
          <td class="align-right">${actionHtml}</td>
        </tr>
      `;
    }).join('');
  }

  window.switchActiveModel = function (modelId) {
    const target = MOCK_DATA.models.find(m => m.id === modelId);
    if (!target) return;

    const current = MOCK_DATA.models.find(m => m.isCurrent);
    if (target.rmse > current.rmse) {
      showToast(`Cannot activate ${target.version}: Validation RMSE (${target.rmse}) is worse than current (${current.rmse}).`, 'warning');
      return;
    }

    MOCK_DATA.models.forEach(m => {
      m.isCurrent = (m.id === modelId);
    });

    showToast(`Active model switched to ${target.version} (${target.name})`, 'success');
    renderModels();
    renderDashboard();
  };

  // Dataset Switcher Listener
  function initDatasetSwitcher() {
    const btnStoreItem = document.getElementById('btn-dataset-store-item');
    const btnRossmann = document.getElementById('btn-dataset-rossmann');

    if (btnStoreItem && btnRossmann) {
      btnStoreItem.onclick = () => {
        state.activeDataset = 'store-item';
        state.selectedSeriesId = 'item-17-store-3';
        btnStoreItem.classList.add('active');
        btnStoreItem.setAttribute('aria-checked', 'true');
        btnRossmann.classList.remove('active');
        btnRossmann.setAttribute('aria-checked', 'false');
        showToast('Switched to Store \u00d7 Item Demand dataset.', 'info');
        renderCurrentRoute();
      };

      btnRossmann.onclick = () => {
        state.activeDataset = 'rossmann';
        state.selectedSeriesId = 'rossmann-store-1';
        btnRossmann.classList.add('active');
        btnRossmann.setAttribute('aria-checked', 'true');
        btnStoreItem.classList.remove('active');
        btnStoreItem.setAttribute('aria-checked', 'false');
        showToast('Switched to Rossmann Store Sales dataset.', 'info');
        renderCurrentRoute();
      };
    }
  }

  // Theme Toggle Listener
  function initThemeToggle() {
    const btnTheme = document.getElementById('theme-toggle');
    if (btnTheme) {
      btnTheme.onclick = () => {
        const nextTheme = state.theme === 'light' ? 'dark' : 'light';
        applyTheme(nextTheme);
        showToast(`Theme switched to ${nextTheme} mode.`, 'info');
      };
    }
  }

  // Application Entry Point
  function initApp() {
    applyTheme(state.theme);
    initDatasetSwitcher();
    initThemeToggle();

    // Global listener for hash navigation
    window.addEventListener('hashchange', handleRoute);

    // Initial route handling
    handleRoute();

    // Refresh button on dashboard
    const btnRefresh = document.getElementById('btn-refresh-dashboard');
    if (btnRefresh) {
      btnRefresh.onclick = () => {
        renderDashboard();
        showToast('Forecasts and safety stock metrics refreshed.', 'success');
      };
    }

    // Expose helpers globally
    window.showToast = showToast;
    window.applyTheme = applyTheme;
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }

})();

