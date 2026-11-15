import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import DisclaimerBanner from './components/DisclaimerBanner';
import Toast from './components/Toast';
import DashboardView from './components/views/DashboardView';
import ItemDetailView from './components/views/ItemDetailView';
import EvaluationView from './components/views/EvaluationView';
import UploadView from './components/views/UploadView';
import ModelsView from './components/views/ModelsView';
import './styles.css';

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'light';
  });

  const [activeRoute, setActiveRoute] = useState(() => {
    const hash = window.location.hash.replace('#/', '');
    return ['dashboard', 'item', 'evaluation', 'upload', 'models'].includes(hash)
      ? hash
      : 'dashboard';
  });

  const [activeDataset, setActiveDataset] = useState('store-item');
  const [selectedSeriesId, setSelectedSeriesId] = useState('item-1-store-1');
  const [toast, setToast] = useState({ message: '', type: 'info' });

  // Sync theme with HTML data-theme attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Sync route with window hash
  useEffect(() => {
    const onHashChange = () => {
      const hash = window.location.hash.replace('#/', '');
      if (['dashboard', 'item', 'evaluation', 'upload', 'models'].includes(hash)) {
        setActiveRoute(hash);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const changeRoute = (route) => {
    setActiveRoute(route);
    window.location.hash = `#/${route}`;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => {
      setToast({ message: '', type: 'info' });
    }, 4500);
  };

  const handleSelectSeries = (seriesId) => {
    setSelectedSeriesId(seriesId);
    changeRoute('item');
  };

  return (
    <div className="app-layout">
      {/* Accessibility Skip Link */}
      <a href="#main-content" className="skip-to-content">
        Skip to main content
      </a>

      {/* Mandatory Domain & Simulation Disclaimers */}
      <DisclaimerBanner />

      {/* Primary Navigation & Dataset Control */}
      <Header
        activeRoute={activeRoute}
        setActiveRoute={changeRoute}
        activeDataset={activeDataset}
        setActiveDataset={setActiveDataset}
        theme={theme}
        setTheme={setTheme}
      />

      {/* Main Content Mount Area */}
      <main id="main-content" className="main-layout" role="main">
        {activeRoute === 'dashboard' && (
          <DashboardView
            activeDataset={activeDataset}
            onSelectSeries={handleSelectSeries}
            theme={theme}
            showToast={showToast}
          />
        )}

        {activeRoute === 'item' && (
          <ItemDetailView
            selectedSeriesId={selectedSeriesId}
            onSelectSeries={setSelectedSeriesId}
            activeDataset={activeDataset}
            theme={theme}
            showToast={showToast}
          />
        )}

        {activeRoute === 'evaluation' && (
          <EvaluationView
            activeDataset={activeDataset}
            theme={theme}
            showToast={showToast}
          />
        )}

        {activeRoute === 'upload' && (
          <UploadView
            activeDataset={activeDataset}
            showToast={showToast}
          />
        )}

        {activeRoute === 'models' && (
          <ModelsView
            showToast={showToast}
          />
        )}
      </main>

      {/* Toast Notification Container */}
      <Toast
        message={toast.message}
        type={toast.type}
        onClose={() => setToast({ message: '', type: 'info' })}
      />
    </div>
  );
}

