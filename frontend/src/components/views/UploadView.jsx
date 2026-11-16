import React, { useState } from 'react';
import { UploadCloud, CheckCircle, AlertTriangle, FileText, Download } from 'lucide-react';
import { uploadCsv } from '../../api/client';

export default function UploadView({ activeDataset, showToast }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [validationResult, setValidationResult] = useState(null);

  const isStoreItem = activeDataset === 'store-item';

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = async (file) => {
    if (!file.name.endsWith('.csv')) {
      showToast('Please upload a valid .csv file.', 'error');
      return;
    }

    setUploading(true);
    setValidationResult(null);

    try {
      const result = await uploadCsv(file, activeDataset);
      setValidationResult(result);
      if (result.valid) {
        showToast(`CSV validated successfully: ${file.name}`, 'success');
      } else {
        showToast(`Validation failed: ${result.errors.length} issue(s) detected`, 'warning');
      }
    } catch (err) {
      showToast(err.message || 'File upload error', 'error');
    } finally {
      setUploading(false);
    }
  };

  const downloadSampleCsv = (isValid = true) => {
    let content = '';
    let filename = '';

    if (isStoreItem) {
      filename = isValid ? 'sample_store_item_valid.csv' : 'sample_store_item_invalid.csv';
      if (isValid) {
        content = `date,store,item,sales
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
        content = `timestamp,store_id,item_code,revenue
2026-13-45,Store_One,NaN,-50.2
2026-10-01,1,1,15
2026-10-01,1,1,15
invalid-date,1,2,twenty-two`;
      }
    } else {
      filename = isValid ? 'sample_rossmann_valid.csv' : 'sample_rossmann_invalid.csv';
      if (isValid) {
        content = `Date,Store,DayOfWeek,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday
2026-10-01,1,4,5200,560,1,1,0,0
2026-10-02,1,5,5600,610,1,1,0,0
2026-10-03,1,6,6100,680,1,0,0,0
2026-10-04,1,7,0,0,0,0,0,0
2026-10-05,1,1,4900,530,1,0,0,0`;
      } else {
        content = `Date,Store,DayOfWeek,Sales
2026-10-01,1,4,-100
2026-10-01,1,4,500`;
      }
    }

    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="page-view active" aria-labelledby="heading-upload">
      <div className="view-header">
        <div>
          <h1 id="heading-upload" className="view-title">
            Data Ingestion &amp; Schema Integrity Gate
          </h1>
          <p className="view-subtitle">
            FR-01 / FR-02: Strict validation against schema, temporal continuity, and outlier boundaries
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => downloadSampleCsv(true)}
            style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
          >
            <Download size={14} /> Download Valid Template
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => downloadSampleCsv(false)}
            style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
          >
            <Download size={14} /> Download Invalid Test CSV
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 1.2fr) minmax(320px, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Upload Zone */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h2 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Upload CSV Sales Records
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
            Target Pipeline: <strong>{isStoreItem ? 'Store Item Demand (Pipeline B)' : 'Rossmann Store Sales (Pipeline A)'}</strong>
          </p>

          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${dragActive ? 'var(--accent)' : 'var(--border)'}`,
              borderRadius: '8px',
              padding: '2.5rem 1.5rem',
              textAlign: 'center',
              background: dragActive ? 'rgba(13, 148, 136, 0.05)' : 'var(--table-header, rgba(0,0,0,0.01))',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
            onClick={() => document.getElementById('csv-file-input').click()}
          >
            <UploadCloud size={40} color="var(--accent)" style={{ margin: '0 auto 0.75rem auto' }} />
            <div style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.25rem' }}>
              Drag and drop your .csv file here, or click to browse
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Supports UTF-8 CSV exports with mandatory headers
            </div>
            <input
              id="csv-file-input"
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleFileInput}
            />
          </div>

          {uploading && (
            <div style={{ marginTop: '1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              Streaming and validating CSV against schema rules...
            </div>
          )}
        </div>

        {/* Schema Reference Panel */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h2 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Enforced Ingestion Schema
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Mandatory columns required to guarantee zero feature leakage
          </p>

          <div style={{ background: 'var(--table-header, rgba(0,0,0,0.02))', padding: '1rem', borderRadius: '8px', fontSize: '0.85rem' }}>
            {isStoreItem ? (
              <ul style={{ paddingLeft: '1.2rem', margin: 0, lineHeight: 1.8 }}>
                <li><code>date</code> &mdash; YYYY-MM-DD contiguous daily timestamp</li>
                <li><code>store</code> &mdash; Integer store ID (1 to 10)</li>
                <li><code>item</code> &mdash; Integer item SKU code (1 to 50)</li>
                <li><code>sales</code> &mdash; Non-negative integer unit sales</li>
              </ul>
            ) : (
              <ul style={{ paddingLeft: '1.2rem', margin: 0, lineHeight: 1.8 }}>
                <li><code>Date</code> &mdash; YYYY-MM-DD contiguous daily timestamp</li>
                <li><code>Store</code> &mdash; Integer store identifier (1 to 1,115)</li>
                <li><code>DayOfWeek</code> &mdash; 1 (Mon) to 7 (Sun)</li>
                <li><code>Sales</code> &mdash; Daily revenue (must be 0 on closed days)</li>
                <li><code>Customers</code> &mdash; Daily foot traffic</li>
                <li><code>Open</code> &mdash; Binary indicator (0 = Closed, 1 = Open)</li>
                <li><code>Promo</code> &mdash; Binary active promotional flag</li>
                <li><code>StateHoliday</code> &mdash; State holiday flag (0, a, b, c)</li>
                <li><code>SchoolHoliday</code> &mdash; School holiday binary flag</li>
              </ul>
            )}
          </div>
        </div>
      </div>

      {/* Validation Report Area */}
      {validationResult && (
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem' }}>
            {validationResult.valid ? (
              <CheckCircle size={22} color="var(--green)" />
            ) : (
              <AlertTriangle size={22} color="var(--red)" />
            )}
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>
              {validationResult.valid ? 'Validation Passed' : 'Validation Failed'}: {validationResult.filename}
            </h3>
          </div>

          <p style={{ fontSize: '0.9rem', color: validationResult.valid ? 'var(--text-muted)' : 'var(--red)', marginBottom: '1rem' }}>
            {validationResult.valid
              ? `Successfully verified ${validationResult.row_count} records against ${isStoreItem ? 'Store Item Demand' : 'Rossmann'} schema constraints.`
              : `File rejected per SRS FR-02 ingestion constraints. Please resolve the following issue(s):`}
          </p>

          {!validationResult.valid && (
            <ul style={{ color: 'var(--red)', fontSize: '0.85rem', lineHeight: 1.6, marginBottom: '1.25rem' }}>
              {validationResult.errors.map((err, idx) => (
                <li key={idx}>{err}</li>
              ))}
            </ul>
          )}

          {/* Sample Preview Table */}
          {validationResult.preview && validationResult.preview.length > 0 && (
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                Parsed Record Preview (First 5 Rows):
              </div>
              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      {Object.keys(validationResult.preview[0]).map((key) => (
                        <th key={key}>{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {validationResult.preview.map((row, idx) => (
                      <tr key={idx}>
                        {Object.values(row).map((val, cIdx) => (
                          <td key={cIdx}>{val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Immediate Forecasts & Reorder Recommendations */}
          {validationResult.predictions && validationResult.predictions.length > 0 && (
            <div style={{ marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <div>
                  <h4 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--blue)', margin: 0 }}>
                    ⚡ Immediate Probabilistic Predictions & Recommended Reorders
                  </h4>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '0.25rem 0 0' }}>
                    Multi-horizon forecasts (P10, P50, P90) and periodic-review (R, S) safety stock generated directly from uploaded data.
                  </p>
                </div>
                <span className="badge badge-blue">
                  {validationResult.predictions.length} Series Forecasted
                </span>
              </div>

              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Series Name</th>
                      <th>Last Data Date</th>
                      <th>Avg Daily Sales</th>
                      <th>Forecast D_T (7d)</th>
                      <th>Safety Stock (SS)</th>
                      <th>Simulated Stock</th>
                      <th>Recommended Q</th>
                    </tr>
                  </thead>
                  <tbody>
                    {validationResult.predictions.map((p, idx) => (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600 }}>{p.name}</td>
                        <td className="font-mono">{p.last_historical_date}</td>
                        <td>{p.avg_historical_sales}</td>
                        <td style={{ fontWeight: 600 }}>{p.forecast_demand_dt}</td>
                        <td>{p.safety_stock}</td>
                        <td>{p.simulated_stock_on_hand}</td>
                        <td>
                          <span className={`badge ${p.recommended_order_q > 0 ? 'badge-amber' : 'badge-green'}`}>
                            {p.recommended_order_q > 0 ? `Order ${p.recommended_order_q} units` : 'Adequate Stock'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

