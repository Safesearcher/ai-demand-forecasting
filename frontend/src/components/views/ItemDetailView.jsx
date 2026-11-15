import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Package, ArrowRight, Save, XCircle } from 'lucide-react';
import { fetchSeries, fetchSeriesDetail, recordDecision } from '../../api/client';
import QuantileForecastChart from '../charts/QuantileForecastChart';

export default function ItemDetailView({
  selectedSeriesId,
  onSelectSeries,
  activeDataset,
  theme,
  showToast,
}) {
  const [seriesList, setSeriesList] = useState([]);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [chartLoading, setChartLoading] = useState(false);

  // Cockpit editable parameters
  const [leadTime, setLeadTime] = useState(3); // L
  const [reviewPeriod, setReviewPeriod] = useState(7); // R
  const [stockOnHand, setStockOnHand] = useState(40); // S
  const [incomingOrders, setIncomingOrders] = useState(0); // I
  const [serviceLevelZ, setServiceLevelZ] = useState(1.65); // z = 1.65 (~95%)
  const [customOrderQty, setCustomOrderQty] = useState('');
  const [decisionReason, setDecisionReason] = useState('');

  // 1. Fetch available series list
  useEffect(() => {
    fetchSeries(activeDataset)
      .then((data) => {
        setSeriesList(data.series || []);
        if (data.series && data.series.length > 0) {
          const currentExists = data.series.some((s) => s.id === selectedSeriesId);
          if (!currentExists) {
            onSelectSeries(data.series[0].id);
          }
        }
      })
      .catch((err) => {
        console.error('Error fetching series list:', err);
      });
  }, [activeDataset]);

  // 2. Fetch series detail & forecasts when selectedSeriesId changes
  useEffect(() => {
    if (!selectedSeriesId) return;
    setChartLoading(true);
    fetchSeriesDetail(selectedSeriesId, activeDataset, 28)
      .then((data) => {
        setDetail(data);
        // Initialize cockpit parameters from selected series metadata if present
        const meta = seriesList.find((s) => s.id === selectedSeriesId);
        if (meta) {
          setLeadTime(meta.lead_time || 3);
          setReviewPeriod(meta.review_period || 7);
          setStockOnHand(meta.stock || 40);
          setIncomingOrders(meta.incoming || 0);
        }
      })
      .catch((err) => {
        console.error('Error fetching series detail:', err);
        showToast('Failed to load forecast for selected series', 'error');
      })
      .finally(() => {
        setChartLoading(false);
        setLoading(false);
      });
  }, [selectedSeriesId, activeDataset, seriesList]);

  // Real-time formula computation (SRS Section 4)
  const L = Number(leadTime);
  const R = Number(reviewPeriod);
  const T = L + R; // Protection Interval
  const S = Number(stockOnHand);
  const I = Number(incomingOrders);
  const z = Number(serviceLevelZ);

  // Daily forecast RMSE (sigma_d) from Phase 4
  const sigmaD = activeDataset === 'store-item' ? 5.7 : 836.2;

  // Forecast demand D_T over T days from forecast records
  let D_T = 0;
  let p90_L = 0;

  if (detail && detail.forecast && detail.forecast.length > 0) {
    const tSlice = detail.forecast.slice(0, Math.min(T, detail.forecast.length));
    D_T = tSlice.reduce((acc, curr) => acc + curr.p50, 0);

    const lSlice = detail.forecast.slice(0, Math.min(L, detail.forecast.length));
    p90_L = lSlice.reduce((acc, curr) => acc + curr.p90, 0);
  } else {
    // Fallback approximation
    D_T = Math.round(12.0 * T);
    p90_L = Math.round(15.0 * L);
  }

  // Exact Safety Stock: SS = z * sigma_d * sqrt(T)
  const exactSS = z * sigmaD * Math.sqrt(T);
  const SS = Math.round(exactSS);

  // Recommended Order: Q = max(0, ceil(D_T + SS - S - I))
  const rawQ = D_T + exactSS - S - I;
  const Q = Math.max(0, Math.ceil(rawQ));

  // Stockout Risk: (S + I) < P90 demand over L
  const isStockoutRisk = (S + I) < p90_L;

  // Status classification
  let statusBadge = {
    class: 'badge-healthy',
    label: 'Healthy Coverage',
    icon: <CheckCircle size={14} />,
  };
  if (isStockoutRisk) {
    statusBadge = {
      class: 'badge-risk',
      label: 'Critical Stockout Risk',
      icon: <AlertTriangle size={14} />,
    };
  } else if (Q > 0) {
    statusBadge = {
      class: 'badge-reorder',
      label: 'Reorder Recommended',
      icon: <AlertTriangle size={14} />,
    };
  }

  // Decision Handlers
  const handleDecision = async (action, quantity) => {
    try {
      const result = await recordDecision({
        series_id: selectedSeriesId,
        action,
        original_quantity: Q,
        recommended_quantity: Q,
        user_adjusted_quantity: quantity !== undefined ? Number(quantity) : null,
        reason: decisionReason || null,
      });
      showToast(
        action === 'accept'
          ? `Order of ${Q} units accepted and recorded.`
          : action === 'adjust'
          ? `Custom replenishment of ${quantity} units recorded.`
          : 'Replenishment recommendation rejected.',
        action === 'reject' ? 'warning' : 'success'
      );
      setCustomOrderQty('');
      setDecisionReason('');
    } catch (err) {
      showToast(err.message || 'Failed to record decision', 'error');
    }
  };

  const selectedMeta = seriesList.find((s) => s.id === selectedSeriesId);

  return (
    <section className="page-view active" aria-labelledby="heading-item">
      <div className="view-header">
        <div>
          <h1 id="heading-item" className="view-title">
            Inventory Decision Support Cockpit
          </h1>
          <p className="view-subtitle">
            Periodic review (R, S) safety stock &amp; dynamic replenishment configuration
          </p>
        </div>

        {/* Series Switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <label htmlFor="series-select" style={{ fontSize: '0.85rem', fontWeight: 600 }}>
            Select Series:
          </label>
          <select
            id="series-select"
            className="form-control"
            style={{ minWidth: 260, padding: '0.4rem 0.75rem' }}
            value={selectedSeriesId || ''}
            onChange={(e) => onSelectSeries(e.target.value)}
          >
            {seriesList.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.category})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Meta Bar */}
      {selectedMeta && (
        <div
          className="glass-panel"
          style={{
            padding: '1rem 1.25rem',
            marginBottom: '1.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem',
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <strong style={{ fontSize: '1.15rem' }}>{selectedMeta.name}</strong>
              <span className={`badge ${statusBadge.class}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                {statusBadge.icon}
                {statusBadge.label}
              </span>
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              {selectedMeta.category} &bull; Pipeline: <strong>{activeDataset === 'store-item' ? 'Store x Item (B)' : 'Rossmann Store Sales (A)'}</strong> &bull; Validation Daily RMSE (&sigma;<sub>d</sub>): <strong>{sigmaD}</strong>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Protection Interval (T = L + R)</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>{T} days</div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>P50 Interval Demand (D_T)</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>{D_T} units</div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Recommended Order (Q)</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 800, color: Q > 0 ? 'var(--amber)' : 'var(--green)' }}>
                {Q} units
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Two Column Layout: Cockpit Parameters on Left, Forecast Chart on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 1fr) minmax(480px, 1.8fr)', gap: '1.5rem', marginBottom: '2rem' }}>
        
        {/* Left Column: Interactive Parameters & Mathematical Engine */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h2 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '1.25rem' }}>
            Interactive Inventory Controls
          </h2>

          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Lead Time (L): {leadTime} days</label>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Supplier fulfillment</span>
            </div>
            <input
              type="range"
              min="1"
              max="21"
              value={leadTime}
              onChange={(e) => setLeadTime(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Review Period (R): {reviewPeriod} days</label>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Order cycle cadence</span>
            </div>
            <input
              type="range"
              min="1"
              max="21"
              value={reviewPeriod}
              onChange={(e) => setReviewPeriod(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>
                Stock on Hand (S)
              </label>
              <input
                type="number"
                className="form-control"
                min="0"
                value={stockOnHand}
                onChange={(e) => setStockOnHand(Number(e.target.value))}
              />
            </div>
            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>
                Incoming Orders (I)
              </label>
              <input
                type="number"
                className="form-control"
                min="0"
                value={incomingOrders}
                onChange={(e) => setIncomingOrders(Number(e.target.value))}
              />
            </div>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>
              Target Non-Stockout Service Level (z)
            </label>
            <select
              className="form-control"
              value={serviceLevelZ}
              onChange={(e) => setServiceLevelZ(Number(e.target.value))}
            >
              <option value="1.28">90% Service Level (z = 1.28)</option>
              <option value="1.65">95% Service Level (z = 1.65 - Standard)</option>
              <option value="1.96">97.5% Service Level (z = 1.96)</option>
              <option value="2.33">99% Service Level (z = 2.33 - Critical)</option>
            </select>
          </div>

          {/* Mathematical Formula Breakdown (SRS Section 4) */}
          <div style={{ background: 'var(--table-header, rgba(0,0,0,0.03))', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
              Formula Verification Engine
            </div>
            <div style={{ fontSize: '0.85rem', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.6 }}>
              <div>SS = z &times; &sigma;<sub>d</sub> &times; &radic;T</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                &nbsp;&nbsp;&nbsp;= {z} &times; {sigmaD} &times; &radic;{T} &approx; <strong>{SS} units</strong>
              </div>
              <div style={{ marginTop: '0.35rem' }}>Q = max(0, &lceil; D<sub>T</sub> + SS - S - I &rceil;)</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                &nbsp;&nbsp;= max(0, &lceil; {D_T} + {SS} - {S} - {I} &rceil;) = <strong style={{ color: 'var(--amber)' }}>{Q} units</strong>
              </div>
              <div style={{ marginTop: '0.35rem' }}>
                Stockout check: S+I ({S + I}) vs P90(L) ({p90_L}) &rarr;{' '}
                <strong style={{ color: isStockoutRisk ? 'var(--red)' : 'var(--green)' }}>
                  {isStockoutRisk ? 'RISK DETECTED' : 'SAFE'}
                </strong>
              </div>
            </div>
          </div>

          {/* Manager Decision Cockpit */}
          <div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.75rem' }}>
              Manager Replenishment Decision
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <button
                type="button"
                className="btn btn-primary"
                style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.35rem' }}
                onClick={() => handleDecision('accept')}
              >
                <CheckCircle size={14} /> Accept Q = {Q}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.35rem' }}
                onClick={() => handleDecision('reject')}
              >
                <XCircle size={14} /> Reject
              </button>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="number"
                className="form-control"
                placeholder="Custom qty..."
                value={customOrderQty}
                onChange={(e) => setCustomOrderQty(e.target.value)}
                style={{ width: '45%' }}
              />
              <button
                type="button"
                className="btn btn-secondary"
                disabled={!customOrderQty}
                onClick={() => handleDecision('adjust', customOrderQty)}
                style={{ flex: 1 }}
              >
                Apply Custom Qty
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Visual Forecast & Quantiles */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h2 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
                Probabilistic Quantile Forecast (P10 / P50 / P90)
              </h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                90-day history with 28-day forward quantile bounds generated by active model
              </p>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <span className="badge badge-model">
                {activeDataset === 'store-item' ? 'LightGBM Multi-Quantile (B)' : 'LightGBM Quantile (A)'}
              </span>
            </div>
          </div>

          <div style={{ flex: 1, minHeight: 380 }}>
            {chartLoading || !detail ? (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ color: 'var(--text-muted)' }}>Loading live model forecast...</span>
              </div>
            ) : (
              <QuantileForecastChart
                history={detail.history}
                forecast={detail.forecast}
                height={380}
                theme={theme}
              />
            )}
          </div>

          <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--table-header, rgba(0,0,0,0.02))', borderRadius: '6px', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
            <span>&bull; P10 (Pessimistic Demand): lower 10th percentile</span>
            <span>&bull; P50 (Expected Demand): median point forecast</span>
            <span>&bull; P90 (Optimistic Demand): 90th percentile stockout threshold</span>
          </div>
        </div>
      </div>
    </section>
  );
}

