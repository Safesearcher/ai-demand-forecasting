import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

export default function HorizonErrorChart({ data = [], height = 280, theme = 'light' }) {
  const canvasRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    const isDark = theme === 'dark';
    const textColor = isDark ? '#94A3B8' : '#64748B';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';

    const labels = data.map((d) => d.day);
    const lgbMAE = data.map((d) => d.lgb_mae);
    const snMAE = data.map((d) => d.seasonal_naive_mae);
    const maMAE = data.map((d) => d.moving_average_mae);

    const ctx = canvasRef.current.getContext('2d');

    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'LightGBM Multi-Quantile (Production)',
            data: lgbMAE,
            borderColor: '#0D9488',
            backgroundColor: '#0D9488',
            borderWidth: 2.5,
            pointRadius: 3,
            tension: 0.1,
          },
          {
            label: 'Seasonal Naive 7-Day Baseline',
            data: snMAE,
            borderColor: '#F59E0B',
            backgroundColor: '#F59E0B',
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 2.5,
            tension: 0.1,
          },
          {
            label: '28-Day Moving Average Baseline',
            data: maMAE,
            borderColor: '#EF4444',
            backgroundColor: '#EF4444',
            borderWidth: 2,
            borderDash: [2, 3],
            pointRadius: 2.5,
            tension: 0.1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: textColor,
              boxWidth: 12,
              font: { family: 'Inter, sans-serif', size: 11 },
            },
          },
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } },
          },
          y: {
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } },
            title: {
              display: true,
              text: 'Mean Absolute Error (MAE)',
              color: textColor,
              font: { family: 'Inter, sans-serif', size: 11 },
            },
          },
        },
      },
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [data, theme]);

  return (
    <div style={{ position: 'relative', width: '100%', height }}>
      <canvas ref={canvasRef} />
    </div>
  );
}

