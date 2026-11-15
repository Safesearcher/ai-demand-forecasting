import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Package, RefreshCw, ArrowUpRight } from 'lucide-react';
import { fetchSeries, fetchSeriesDetail } from '../../api/client';
import QuantileForecastChart from '../charts/QuantileForecastChart';

export default function DashboardView({
  activeDataset,
  onSelectSeries,
  theme,
  showToast,
}) {
  const [seriesList, setSeriesList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [selectedSeriesId, setSelectedSeriesId] = useState(null);
  const [previewDetail, setPreviewDetail] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const loadSeries = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchSeries(activeDataset);
      setSeriesList(data.series || []);
      if (data.series && data.series.length > 0) {
        setSelectedSeriesId(data.series[0].id);
      }
    } catch (err) {
      setError(err.message || 'Failed to load series data from backend API.');
      showToast(err.message || 'Error connecting to backend API', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSeries();
  }, [activeDataset]);

  // Load preview chart when selectedSeriesId changes
  useEffect(() => {
    if (!selectedSeriesId) return;
    let isCancelled = false;
    setPreviewLoading(true);
    fetchSeriesDetail(selectedSeriesId, activeDataset, 28)
      .then((detail) => {
        if (!isCancelled) {
          setPreviewDetail(detail);
        }
      })
      .catch((err) => {
        console.error('Failed to load series detail preview:', err);
      })
      .finally(() => {
        if (!isCancelled) setPreviewLoading(false);
      });
    return () => {
      isCancelled = true;
    };
  }, [selectedSeriesId, activeDataset]);

  // KPI Calculations
  const totalCount = seriesList.length;
  const criticalCount = seriesList.filter((s) => s.status === 'critical').length;
  const reorderCount = seriesList.filter((s) => s.status === 'reorder').length;
  const healthyCount = seriesList.filter((s) => s.status === 'healthy').length;

  const filteredSeries = seriesList.filter((s) => {
    if (filterStatus === 'all') return true;
    return s.status === filterStatus;
  });

  const selectedSeries = seriesList.find((s) => s.id === selectedSeriesId);

  return (
    <section className="page-view active" aria-labelledby="heading-dashboard">
      <div className="view-header">
        <div>
          <h1 id="heading-dashboard" className="view-title">
            Executive Decision Dashboard
          </h1>
          <p className="view-subtitle">
            Real-time quantile demand forecasts &amp; periodic review replenishment decisions
          </p>
        </div>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={loadSeries}
          disabled={loading}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="validation-card invalid" style={{ marginBottom: '1.5rem' }}>
          <div className="validation-header">
            <AlertTriangle size={20} color="var(--red)" />
            <span className="validation-title">Backend Connection Issue</span>
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--red-text)', margin: '0.5rem 0' }}>
            {error}
          </p>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Ensure FastAPI server is running: <code>.venv/bin/uvicorn src.api:app --port 8000</code>
          </p>
        </div>
      )}

      {/* KPI Stats Strip */}
      <div className="kpi-grid">
        <div className="kpi-card glass-panel">
          <div className="kpi-label">Monitored Series</div>
          <div className="kpi-value">{loading ? '...' : totalCount}</div>
          <div className="kpi-subtext">Active items / stores in pipeline</div>
        </div>

        <div className="kpi-card glass-panel kpi-critical">
          <div className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <AlertTriangle size={14} color="var(--red)" />
            Critical Stockout Risk
          </div>
          <div className="kpi-value" style={{ color: 'var(--red)' }}>
            {loading ? '...' : criticalCount}
          </div>
          <div className="kpi-subtext">Net stock &lt; P90 lead-time demand</div>
        </div>

        <div className="kpi-card glass-panel kpi-reorder">
          <div className="kpi-label">Reorder Recommended</div>
          <div className="kpi-value" style={{ color: 'var(--amber)' }}>
            {loading ? '...' : reorderCount}
          </div>
          <div className="kpi-subtext">Q &gt; 0 to satisfy protection interval</div>
        </div>

        <div className="kpi-card glass-panel kpi-healthy">
          <div className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <CheckCircle size={14} color="var(--green)" />
            Healthy Coverage
          </div>
          <div className="kpi-value" style={{ color: 'var(--green)' }}>
            {loading ? '...' : healthyCount}
          </div>
          <div className="kpi-subtext">Sfficient inventory buffer</div>
        </div>
      </div>

      {/* Quick Visual Forecast Preview Card */}
      {selectedSeries && previewDetail && (
        <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div>
              <span className="badge badge-model" style={{ marginRight: '0.5rem' }}>
                {activeDataset === 'store-item' ? 'Store x Item Pipeline B' : 'Rossmann Pipeline A'}
              </span>
              <strong style={{ fontSize: '1.05rem' }}>{selectedSeries.name}</strong>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginLeft: '0.75rem' }}>
                Historical observations &amp; 28-day LightGBM probabilistic quantiles (P10/P50/P90)
              </span>
            </div>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onSelectSeries(selectedSeries.id)}
              style={{ fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
            >
              Open in Inventory Cockpit <ArrowUpRight size={14} />
            </button>
          </div>
          {previewLoading ? (
            <div style={{ height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ color: 'var(--text-muted)' }}>Generating probabilistic forecast...</span>
            </div>
          ) : (
            <QuantileForecastChart
              history={previewDetail.history}
              forecast={previewDetail.forecast}
              height={280}
              theme={theme}
            />
          )}
        </div>
      )}

      {/* Filter and Data Table Section */}
      <div className="glass-panel" style={{ padding: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div className="filter-button-group" role="group" aria-label="Status Filter">
            <button
              type="button"
              className={`filter-btn ${filterStatus === 'all' ? 'active' : ''}`}
              onClick={() => setFilterStatus('all')}
            >
              All Series ({totalCount})
            </button>
            <button
              type="button"
              className={`filter-btn ${filterStatus === 'critical' ? 'active' : ''}`}
              onClick={() => setFilterStatus('critical')}
            >
              Critical Risk ({criticalCount})
            </button>
            <button
              type="button"
              className={`filter-btn ${filterStatus === 'reorder' ? 'active' : ''}`}
              onClick={() => setFilterStatus('reorder')}
            >
              Reorder ({reorderCount})
            </button>
            <button
              type="button"
              className={`filter-btn ${filterStatus === 'healthy' ? 'active' : ''}`}
              onClick={() => setFilterStatus('healthy')}
            >
              Healthy ({healthyCount})
            </button>
          </div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Showing {filteredSeries.length} of {totalCount} series
          </span>
        </div>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Series Identifier</th>
                <th>Classification</th>
                <th style={{ textAlign: 'right' }}>Stock on Hand (S)</th>
                <th style={{ textAlign: 'right' }}>Incoming (I)</th>
                <th style={{ textAlign: 'right' }}>L / R (Days)</th>
                <th style={{ textAlign: 'right' }}>Forecast (D_T)</th>
                <th style={{ textAlign: 'right' }}>Safety Stock (SS)</th>
                <th style={{ textAlign: 'right' }}>Order Qty (Q)</th>
                <th style={{ textAlign: 'center' }}>Inventory Status</th>
                <th style={{ textAlign: 'center' }}>Cockpit</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={10} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    Loading series and evaluating reorder thresholds...
                  </td>
                </tr>
              ) : filteredSeries.length === 0 ? (
                <tr>
                  <td colSpan={10} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    No series found matching the selected filter.
                  </td>
                </tr>
              ) : (
                filteredSeries.map((s) => {
                  const isSelected = s.id === selectedSeriesId;
                  return (
                    <tr
                      key={s.id}
                      onClick={() => setSelectedSeriesId(s.id)}
                      style={{
                        cursor: 'pointer',
                        background: isSelected ? 'var(--table-row-selected, rgba(13, 148, 136, 0.08))' : undefined,
                      }}
                    >
                      <td>
                        <strong>{s.name}</strong>
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{s.category}</td>
                      <td style={{ textAlign: 'right' }}>{s.stock}</td>
                      <td style={{ textAlign: 'right' }}>{s.incoming}</td>
                      <td style={{ textAlign: 'right' }}>{s.lead_time} / {s.review_period}</td>
                      <td style={{ textAlign: 'right', fontWeight: 600 }}>{s.forecast_demand_dt}</td>
                      <td style={{ textAlign: 'right' }}>{s.safety_stock}</td>
                      <td style={{ textAlign: 'right', fontWeight: 700, color: s.order_quantity > 0 ? 'var(--amber)' : 'inherit' }}>
                        {s.order_quantity}
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        <span className={`badge badge-${s.status}`}>
                          {s.status_label}
                        </span>
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectSeries(s.id);
                          }}
                        >
                          Configure
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

