import React, { useState, useEffect } from 'react';
import { BarChart2, ShieldCheck, TrendingUp, RefreshCw } from 'lucide-react';
import { fetchEvaluation } from '../../api/client';
import HorizonErrorChart from '../charts/HorizonErrorChart';

export default function EvaluationView({ activeDataset, theme, showToast }) {
  const [pipeline, setPipeline] = useState(activeDataset === 'rossmann' ? 'A' : 'B');
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Sync pipeline with activeDataset prop
  useEffect(() => {
    setPipeline(activeDataset === 'rossmann' ? 'A' : 'B');
  }, [activeDataset]);

  const loadEvaluation = async (pipe) => {
    setLoading(true);
    try {
      const data = await fetchEvaluation(pipe);
      setEvalData(data);
    } catch (err) {
      console.error('Failed to load evaluation data:', err);
      showToast(err.message || 'Error loading Phase 4 evaluation data', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEvaluation(pipeline);
  }, [pipeline]);

  return (
    <section className="page-view active" aria-labelledby="heading-evaluation">
      <div className="view-header">
        <div>
          <h1 id="heading-evaluation" className="view-title">
            Empirical Evaluation &amp; Model Benchmarking
          </h1>
          <p className="view-subtitle">
            Phase 4 contiguous chronological holdout validation, quantile calibration &amp; stockout classification
          </p>
        </div>

        {/* Pipeline Toggle */}
        <div className="filter-button-group" role="group" aria-label="Pipeline Toggle">
          <button
            type="button"
            className={`filter-btn ${pipeline === 'B' ? 'active' : ''}`}
            onClick={() => setPipeline('B')}
          >
            Pipeline B (Store Item Demand)
          </button>
          <button
            type="button"
            className={`filter-btn ${pipeline === 'A' ? 'active' : ''}`}
            onClick={() => setPipeline('A')}
          >
            Pipeline A (Rossmann Sales)
          </button>
        </div>
      </div>

      {loading ? (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <RefreshCw size={24} className="spin" style={{ margin: '0 auto 1rem auto' }} />
          <div>Loading empirical Phase 4 holdout evaluation metrics...</div>
        </div>
      ) : !evalData ? (
        <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: 'var(--red)' }}>
          Failed to load evaluation metrics.
        </div>
      ) : (
        <>
          {/* Section 1: Main Multi-Quantile Benchmark Table */}
          <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div>
                <h2 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
                  Quantile Forecasting Performance vs. Baselines
                </h2>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Evaluated on contiguous chronological test holdout (Pipeline {pipeline}: {pipeline === 'B' ? '90 days' : '42 days'})
                </span>
              </div>
              <span className="badge badge-healthy" style={{ fontSize: '0.75rem' }}>
                Zero Lookahead Leakage
              </span>
            </div>

            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model Family</th>
                    <th style={{ textAlign: 'right' }}>MAE</th>
                    <th style={{ textAlign: 'right' }}>RMSE</th>
                    <th style={{ textAlign: 'right' }}>WMAPE (%)</th>
                    <th style={{ textAlign: 'right' }}>Bias</th>
                    <th style={{ textAlign: 'right' }}>Pinball P10</th>
                    <th style={{ textAlign: 'right' }}>Pinball P50</th>
                    <th style={{ textAlign: 'right' }}>Pinball P90</th>
                    <th style={{ textAlign: 'right' }}>Coverage (Target: 80%)</th>
                    <th style={{ textAlign: 'right' }}>Mean Interval Width</th>
                  </tr>
                </thead>
                <tbody>
                  {evalData.metrics.map((m, idx) => {
                    const isChampion = m.model === 'lgbm';
                    return (
                      <tr
                        key={idx}
                        style={{
                          background: isChampion ? 'var(--table-row-selected, rgba(13, 148, 136, 0.08))' : undefined,
                          fontWeight: isChampion ? 600 : 'normal',
                        }}
                      >
                        <td>
                          <strong>{m.model === 'lgbm' ? 'LightGBM Multi-Quantile (Production)' : m.model === 'mlp' ? 'PyTorch QuantileMLP' : m.model.replace(/_/g, ' ').toUpperCase()}</strong>
                          {isChampion && (
                            <span className="badge badge-model" style={{ marginLeft: '0.5rem', fontSize: '0.65rem' }}>
                              Champion
                            </span>
                          )}
                        </td>
                        <td style={{ textAlign: 'right' }}>{m.mae}</td>
                        <td style={{ textAlign: 'right' }}>{m.rmse}</td>
                        <td style={{ textAlign: 'right', color: isChampion ? 'var(--green)' : 'inherit' }}>
                          {m.wmape}%
                        </td>
                        <td style={{ textAlign: 'right' }}>{m.bias}</td>
                        <td style={{ textAlign: 'right' }}>{m.pinball_p10}</td>
                        <td style={{ textAlign: 'right' }}>{m.pinball_p50}</td>
                        <td style={{ textAlign: 'right' }}>{m.pinball_p90}</td>
                        <td style={{ textAlign: 'right', fontWeight: 700, color: Math.abs(m.coverage_p10_p90 - 80) < 5 ? 'var(--green)' : 'inherit' }}>
                          {m.coverage_p10_p90}%
                        </td>
                        <td style={{ textAlign: 'right' }}>{m.mean_interval_width}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 2: Horizon Degradation Chart & Stockout Classification Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(350px, 1fr) minmax(350px, 1fr)', gap: '1.5rem', marginBottom: '1.75rem' }}>
            
            {/* Horizon Error Progression */}
            <div className="glass-panel" style={{ padding: '1.25rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <h2 style={{ fontSize: '1rem', fontWeight: 700 }}>
                  Forecast Error by Horizon (Days 1 to 14)
                </h2>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Error progression across multi-step direct forecast windows
                </span>
              </div>
              <HorizonErrorChart data={evalData.horizon_error} height={260} theme={theme} />
            </div>

            {/* Stockout Alert Classification */}
            <div className="glass-panel" style={{ padding: '1.25rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <h2 style={{ fontSize: '1rem', fontWeight: 700 }}>
                  Stockout Alert Decision Classification
                </h2>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Binary alerting capability under simulated stockout conditions
                </span>
              </div>

              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th style={{ textAlign: 'right' }}>Precision</th>
                      <th style={{ textAlign: 'right' }}>Recall</th>
                      <th style={{ textAlign: 'right' }}>F1-Score</th>
                      <th style={{ textAlign: 'right' }}>AUC-ROC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalData.classification.map((c, idx) => (
                      <tr key={idx}>
                        <td><strong>{c.model.toUpperCase()}</strong></td>
                        <td style={{ textAlign: 'right' }}>{c.precision}</td>
                        <td style={{ textAlign: 'right' }}>{c.recall}</td>
                        <td style={{ textAlign: 'right', fontWeight: 700 }}>{c.f1}</td>
                        <td style={{ textAlign: 'right', color: 'var(--accent)' }}>{c.auc_roc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ marginTop: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Stockout alert policy triggers when <code>S + I &lt; P90 demand over lead time L</code>. LightGBM demonstrates robust sensitivity with high AUC-ROC, ensuring stockouts are proactively flagged before depletion occurs.
              </div>
            </div>
          </div>

          {/* Section 3: Subgroup Fairness Breakdown */}
          {evalData.subgroups && evalData.subgroups.length > 0 && (
            <div className="glass-panel" style={{ padding: '1.25rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <h2 style={{ fontSize: '1rem', fontWeight: 700 }}>
                  Subgroup Algorithmic Fairness &amp; Error Distribution
                </h2>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Audited performance across store categories, assortments, and promotions
                </span>
              </div>

              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Subgroup Dimension</th>
                      <th>Cohort Value</th>
                      <th style={{ textAlign: 'right' }}>Sample Instances</th>
                      <th style={{ textAlign: 'right' }}>Cohort MAE</th>
                      <th style={{ textAlign: 'right' }}>Cohort WMAPE (%)</th>
                      <th style={{ textAlign: 'right' }}>Empirical Coverage (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalData.subgroups.slice(0, 10).map((sg, idx) => (
                      <tr key={idx}>
                        <td><strong>{sg.subgroup}</strong></td>
                        <td><span className="badge badge-model">{sg.value}</span></td>
                        <td style={{ textAlign: 'right' }}>{sg.count.toLocaleString()}</td>
                        <td style={{ textAlign: 'right' }}>{sg.mae}</td>
                        <td style={{ textAlign: 'right' }}>{sg.wmape}%</td>
                        <td style={{ textAlign: 'right' }}>{sg.coverage}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}

