/**
 * AI Demand Forecasting and Inventory Decision Support
 * Liquid Glass UI & Liquid Logo Module
 * 
 * References & Credits:
 * - https://github.com/dashersw/liquid-glass-js (Apple-inspired glass specular follow)
 * - https://github.com/collidingScopes/liquid-logo (Interactive fluid distortion on brand marks)
 */

(function () {
  'use strict';

  // Check URL params for lite mode (?lite=1)
  const urlParams = new URLSearchParams(window.location.search);
  const isLiteMode = urlParams.get('lite') === '1';

  // =========================================================================
  // 1. LIQUID GLASS SPECULAR HIGHLIGHT TRACKER
  // =========================================================================
  function initLiquidGlass() {
    if (isLiteMode) return;

    // Selector for interactive glass panels
    const glassSelector = '.kpi-card, .glass-panel, .modal-dialog, .decision-callout, .dataset-switcher, .card';

    function attachGlassListener(el) {
      if (el.dataset.hasGlassListener) return;
      el.dataset.hasGlassListener = 'true';

      el.addEventListener('mousemove', (e) => {
        const rect = el.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        el.style.setProperty('--mouse-x', `${x}px`);
        el.style.setProperty('--mouse-y', `${y}px`);
      }, { passive: true });

      el.addEventListener('mouseleave', () => {
        el.style.setProperty('--mouse-x', `-999px`);
        el.style.setProperty('--mouse-y', `-999px`);
      }, { passive: true });
    }

    // Attach to existing elements
    document.querySelectorAll(glassSelector).forEach(attachGlassListener);

    // MutationObserver to attach to dynamically created elements
    const observer = new MutationObserver((mutations) => {
      mutations.forEach(m => {
        m.addedNodes.forEach(node => {
          if (node.nodeType === 1) {
            if (node.matches && node.matches(glassSelector)) attachGlassListener(node);
            if (node.querySelectorAll) node.querySelectorAll(glassSelector).forEach(attachGlassListener);
          }
        });
      });
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  // =========================================================================
  // 2. LIQUID LOGO: Rising Demand Curve "A" Mark with Fluid Refraction
  // =========================================================================
  function initLiquidLogo() {
    const canvas = document.getElementById('logo-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 36;
    const height = 36;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    let rippleStrength = 0.0;
    let targetRipple = 0.0;
    let time = 0;
    let isHovered = false;

    // Hover triggers fluid agitation
    const brandWrap = canvas.closest('.brand-section') || canvas;
    brandWrap.addEventListener('mouseenter', () => {
      isHovered = true;
      targetRipple = 1.0;
    });

    brandWrap.addEventListener('mouseleave', () => {
      isHovered = false;
      targetRipple = 0.0;
    });

    function drawGeometricLogo(w, h, t, ripple) {
      ctx.clearRect(0, 0, w, h);

      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const primaryColor = isDark ? '#FACC15' : '#b7791f';
      const secondaryColor = isDark ? '#EAB308' : '#9a6516';
      const glowColor = isDark ? 'rgba(250, 204, 21, 0.4)' : 'rgba(183, 121, 31, 0.25)';

      // Background rounded container with subtle glass refraction
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(1, 1, w - 2, h - 2, 8);
      ctx.fillStyle = isDark ? 'rgba(19, 27, 46, 0.85)' : 'rgba(241, 245, 249, 0.9)';
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = isDark ? 'rgba(45, 212, 191, 0.3)' : 'rgba(13, 148, 136, 0.25)';
      ctx.stroke();

      // Fluid Wave distortion on the curve
      // The mark is an iconic "A" formed by an upward exponential demand curve & safety crossbar
      const wave = Math.sin(t * 3.5) * (1.2 + ripple * 2.5);

      ctx.beginPath();
      // Left leg rising curve
      ctx.moveTo(7, 28);
      ctx.quadraticCurveTo(12 + wave * 0.4, 22, 18 + wave * 0.2, 7);
      // Right leg descending curve
      ctx.quadraticCurveTo(24 - wave * 0.3, 19, 29, 28);
      
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = primaryColor;
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 6 + ripple * 6;
      ctx.stroke();

      // Crossbar: represents the P50 median safety stock threshold
      ctx.beginPath();
      const barY = 20 + Math.sin(t * 4.0) * (0.8 + ripple * 1.5);
      ctx.moveTo(11.5, barY);
      ctx.lineTo(24.5, barY);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = secondaryColor;
      ctx.stroke();

      // Uncertainty Apex Particle (at peak of A curve)
      ctx.beginPath();
      const apexX = 18 + wave * 0.2;
      const apexY = 7;
      ctx.arc(apexX, apexY, 2.2 + ripple * 0.8, 0, Math.PI * 2);
      ctx.fillStyle = isDark ? '#FFFFFF' : primaryColor;
      ctx.shadowBlur = 8;
      ctx.shadowColor = primaryColor;
      ctx.fill();

      ctx.restore();
    }

    function renderLogo() {
      time += 0.03;
      rippleStrength += (targetRipple - rippleStrength) * 0.1;
      drawGeometricLogo(width, height, time, rippleStrength);
      requestAnimationFrame(renderLogo);
    }

    renderLogo();
  }

  // Initialize once DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initLiquidGlass();
      initLiquidLogo();
    });
  } else {
    initLiquidGlass();
    initLiquidLogo();
  }

  // Export helpers
  window.initLiquidGlass = initLiquidGlass;

})();

