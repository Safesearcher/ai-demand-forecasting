import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle, RefreshCw, Layers } from 'lucide-react';
import { fetchModels, switchModel } from '../../api/client';

export default function ModelsView({ showToast }) {
  const [modelData, setModelData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);

  const loadModels = async () => {
    setLoading(true);
    try {
      const data = await fetchModels();
      setModelData(data);
    } catch (err) {
      console.error('Failed to load models:', err);
      showToast('Error loading models registry', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleSwitch = async (pipeline, modelId) => {
    setSwitching(true);
    try {
      const res = await switchModel(pipeline, modelId);
      showToast(res.message || 'Model activated successfully', 'success');
      await loadModels();
    } catch (err) {
      showToast(err.message || 'Failed to switch active model', 'error');
    } finally {
      setSwitching(false);
    }
  };

  return (
    <section className="page-view active" aria-labelledby="heading-models">
      <div className="view-header">
        <div>
          <h1 id="heading-models" className="view-title">
            Model Registry &amp; Lifecycle Management
          </h1>
          <p className="view-subtitle">
            FR-13: Version control, validation gates, and dynamic model switching in production
          </p>
        </div>

        <button
          type="button"
          className="btn btn-secondary"
          onClick={loadModels}
          disabled={loading}
          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}
        >
          <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh Registry
        </button>
      </div>

      {loading ? (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <RefreshCw size={24} className="spin" style={{ margin: '0 auto 1rem auto' }} />
          <div>Querying model repository and active configuration...</div>
        </div>
      ) : !modelData ? (
        <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: 'var(--red)' }}>
          Failed to load model registry.
        </div>
      ) : (
        <>
          <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1.75rem' }}>
            <div style={{ marginBottom: '1rem' }}>
              <h2 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
                Registered Model Estimators
              </h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Switching active version redirects live real-time inference across both pipelines
              </span>
            </div>

            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model / Version</th>
                    <th>Architecture</th>
                    <th>Pipeline</th>
                    <th style={{ textAlign: 'right' }}>Holdout RMSE</th>
                    <th style={{ textAlign: 'right' }}>Holdout MAE</th>
                    <th style={{ textAlign: 'right' }}>WMAPE (%)</th>
                    <th style={{ textAlign: 'right' }}>Coverage (%)</th>
                    <th style={{ textAlign: 'center' }}>Status</th>
                    <th style={{ textAlign: 'center' }}>Production Action</th>
                  </tr>
                </thead>
                <tbody>
                  {modelData.models.map((m) => (
                    <tr
                      key={m.id}
                      style={{
                        background: m.is_current ? 'var(--table-row-selected, rgba(13, 148, 136, 0.08))' : undefined,
                      }}
                    >
                      <td>
                        <strong>{m.name}</strong>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{m.version}</div>
                      </td>
                      <td style={{ fontSize: '0.85rem' }}>{m.architecture}</td>
                      <td>
                        <span className="badge badge-model">
                          Pipeline {m.pipeline}
                        </span>
                      </td>
                      <td style={{ textAlign: 'right' }}>{m.rmse}</td>
                      <td style={{ textAlign: 'right' }}>{m.mae}</td>
                      <td style={{ textAlign: 'right', fontWeight: 600 }}>{m.wmape}%</td>
                      <td style={{ textAlign: 'right' }}>{m.coverage}%</td>
                      <td style={{ textAlign: 'center' }}>
                        {m.is_current ? (
                          <span className="badge badge-healthy">Active Model</span>
                        ) : m.status === 'validated' ? (
                          <span className="badge badge-reorder">Validated Candidate</span>
                        ) : (
                          <span className="badge badge-model">Benchmark</span>
                        )}
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        {m.is_current ? (
                          <span style={{ fontSize: '0.8rem', color: 'var(--green)', fontWeight: 600 }}>
                            &#10004; Active in Production
                          </span>
                        ) : (
                          <button
                            type="button"
                            className="btn btn-primary"
                            disabled={switching}
                            style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
                            onClick={() => handleSwitch(m.pipeline, m.id)}
                          >
                            Switch to Active
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Architecture Deep-Dive Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
            <div className="glass-panel" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <Layers size={18} color="var(--accent)" />
                <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>
                  LightGBM Multi-Quantile Architecture
                </h3>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                Production-grade ensemble of three distinct gradient boosted regressors optimized via pinball quantile loss (&alpha; &isin; &#123;0.1, 0.5, 0.9&#125;). Monotonic quantile sorting is strictly applied across the predicted columns to guarantee that P10 &le; P50 &le; P90 without quantile boundary crossing.
              </p>
            </div>

            <div className="glass-panel" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <Cpu size={18} color="var(--accent)" />
                <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>
                  PyTorch QuantileMLP Architecture
                </h3>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                Deep neural network featuring entity embedding layers for discrete series identifiers, dense fully connected layers with dropout regularization, and a non-crossing softplus parameterization head: <code>P10 = base</code>, <code>P50 = P10 + Softplus(&Delta;1)</code>, and <code>P90 = P50 + Softplus(&Delta;2)</code>.
              </p>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

