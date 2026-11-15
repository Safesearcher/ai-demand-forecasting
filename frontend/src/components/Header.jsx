import React from 'react';
import { LayoutDashboard, Package, BarChart2, UploadCloud, Cpu, Sun, Moon } from 'lucide-react';

export default function Header({
  activeRoute,
  setActiveRoute,
  activeDataset,
  setActiveDataset,
  theme,
  setTheme,
}) {
  const toggleTheme = () => {
    const nextTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
  };

  return (
    <header className="app-header" role="banner">
      <div className="header-container">
        <div className="header-top-row">
          <a
            href="#/dashboard"
            onClick={(e) => {
              e.preventDefault();
              setActiveRoute('dashboard');
            }}
            className="brand-section"
            aria-label="Home — AI Demand Forecasting & Inventory Decision Support"
          >
            <div className="brand-icon-wrap" aria-hidden="true">
              <svg width="28" height="28" viewBox="0 0 36 36" fill="none">
                <path
                  d="M6 28 C11 22, 18 10, 18 7 C18 10, 25 22, 30 28"
                  stroke="#FACC15"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                />
                <line x1="10" y1="20" x2="26" y2="20" stroke="#FDE047" strokeWidth="2.5" strokeLinecap="round" />
                <circle cx="18" cy="7" r="2.5" fill="#FFFFFF" />
              </svg>
            </div>
            <div className="brand-title-wrap">
              <span className="brand-name">AI Demand Forecasting &amp; Inventory Decision Support</span>
              <span className="brand-subtitle">Quantile Uncertainty (P10/P50/P90) &bull; Safety Stock Optimization</span>
            </div>
          </a>

          <div className="header-controls">
            <div className="dataset-segmented-control" role="radiogroup" aria-label="Active Dataset">
              <button
                type="button"
                className={`dataset-segment-btn ${activeDataset === 'store-item' ? 'active' : ''}`}
                onClick={() => setActiveDataset('store-item')}
                role="radio"
                aria-checked={activeDataset === 'store-item'}
                title="Kaggle Store Item Demand (10 stores x 50 items)"
              >
                Store &times; Item Demand
              </button>
              <button
                type="button"
                className={`dataset-segment-btn ${activeDataset === 'rossmann' ? 'active' : ''}`}
                onClick={() => setActiveDataset('rossmann')}
                role="radio"
                aria-checked={activeDataset === 'rossmann'}
                title="Kaggle Rossmann Store Sales (1,115 stores with promo/holiday)"
              >
                Rossmann Store Sales
              </button>
            </div>

            <button
              type="button"
              className="btn-icon"
              onClick={toggleTheme}
              aria-label="Toggle Theme"
              title="Toggle color scheme"
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>

        <nav className="app-nav" role="navigation" aria-label="Main Navigation">
          <button
            type="button"
            className={`nav-link ${activeRoute === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveRoute('dashboard')}
          >
            <LayoutDashboard size={16} />
            Dashboard
          </button>
          <button
            type="button"
            className={`nav-link ${activeRoute === 'item' ? 'active' : ''}`}
            onClick={() => setActiveRoute('item')}
          >
            <Package size={16} />
            Item Detail
          </button>
          <button
            type="button"
            className={`nav-link ${activeRoute === 'evaluation' ? 'active' : ''}`}
            onClick={() => setActiveRoute('evaluation')}
          >
            <BarChart2 size={16} />
            Evaluation
          </button>
          <button
            type="button"
            className={`nav-link ${activeRoute === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveRoute('upload')}
          >
            <UploadCloud size={16} />
            Data Upload
          </button>
          <button
            type="button"
            className={`nav-link ${activeRoute === 'models' ? 'active' : ''}`}
            onClick={() => setActiveRoute('models')}
          >
            <Cpu size={16} />
            Models
          </button>
        </nav>
      </div>
    </header>
  );
}

