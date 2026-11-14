/**
 * AI Demand Forecasting and Inventory Decision Support
 * Smooth Transitions & Micro-Interaction System
 * References: View Transitions API, Emil Kowalski design engineering principles
 */

(function () {
  'use strict';

  const urlParams = new URLSearchParams(window.location.search);
  const isLiteMode = urlParams.get('lite') === '1';
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // =========================================================================
  // 1. PAGE INITIALIZATION (IMMEDIATE LOAD)
  // =========================================================================
  function revealPageContent() {
    document.body.classList.add('app-ready');
    triggerKpiCountUps();
  }

  // =========================================================================
  // 2. TABULAR KPI NUMBER COUNT-UP ANIMATION
  // =========================================================================
  function animateCounter(el, targetVal, durationMs = 850) {
    if (!el) return;
    const isNum = !isNaN(Number(targetVal.toString().replace(/,/g, '')));
    if (!isNum) {
      el.textContent = targetVal;
      return;
    }

    const end = parseFloat(targetVal.toString().replace(/,/g, ''));
    if (prefersReducedMotion || isLiteMode) {
      el.textContent = end.toLocaleString();
      return;
    }

    const startTime = performance.now();
    function tick(now) {
      const progress = Math.min((now - startTime) / durationMs, 1.0);
      // Ease out expo
      const ease = progress === 1.0 ? 1.0 : 1 - Math.pow(2, -10 * progress);
      const current = Math.round(end * ease);
      el.textContent = current.toLocaleString();
      if (progress < 1.0) {
        requestAnimationFrame(tick);
      } else {
        el.textContent = end.toLocaleString();
      }
    }
    requestAnimationFrame(tick);
  }

  function triggerKpiCountUps() {
    const kpiAttention = document.getElementById('kpi-attention-count');
    const kpiUnits = document.getElementById('kpi-units-to-order');
    const kpiStockout = document.getElementById('kpi-stockout-count');

    if (kpiAttention && kpiAttention.textContent) {
      animateCounter(kpiAttention, kpiAttention.textContent, 650);
    }
    if (kpiUnits && kpiUnits.textContent) {
      animateCounter(kpiUnits, kpiUnits.textContent, 950);
    }
    if (kpiStockout && kpiStockout.textContent) {
      animateCounter(kpiStockout, kpiStockout.textContent, 650);
    }
  }

  // =========================================================================
  // 3. MAGNETIC BUTTONS INTERACTION
  // =========================================================================
  function initMagneticButtons() {
    if (isLiteMode || prefersReducedMotion) return;

    // Apply to primary action buttons
    document.querySelectorAll('.btn-primary, .btn-magnetic').forEach(btn => {
      btn.addEventListener('mousemove', (e) => {
        const rect = btn.getBoundingClientRect();
        const x = e.clientX - (rect.left + rect.width / 2);
        const y = e.clientY - (rect.top + rect.height / 2);
        btn.style.transform = `translate(${x * 0.15}px, ${y * 0.15}px)`;
      });

      btn.addEventListener('mouseleave', () => {
        btn.style.transform = '';
      });
    });
  }

  // =========================================================================
  // 4. ROUTE VIEW TRANSITIONS
  // =========================================================================
  function performViewTransition(renderCallback) {
    if (document.startViewTransition && !prefersReducedMotion && !isLiteMode) {
      try {
        const transition = document.startViewTransition(() => {
          renderCallback();
        });
        if (transition) {
          if (transition.ready) transition.ready.catch(() => {});
          if (transition.finished) transition.finished.catch(() => {});
          if (transition.updateCallbackDone) transition.updateCallbackDone.catch(() => {});
        }
      } catch (e) {
        renderCallback();
      }
    } else {
      // CSS Fallback
      const main = document.getElementById('main-content');
      if (main) {
        main.classList.add('route-transition-out');
        setTimeout(() => {
          renderCallback();
          main.classList.remove('route-transition-out');
          main.classList.add('route-transition-in');
          setTimeout(() => main.classList.remove('route-transition-in'), 350);
        }, 120);
      } else {
        renderCallback();
      }
    }
  }

  // Expose global motion helpers
  window.appMotion = {
    performViewTransition,
    animateCounter,
    triggerKpiCountUps,
    initMagneticButtons
  };

  // Run on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      revealPageContent();
      initMagneticButtons();
    });
  } else {
    revealPageContent();
    initMagneticButtons();
  }

})();

