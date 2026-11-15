import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

export default function QuantileForecastChart({
  history = [],
  forecast = [],
  height = 360,
  theme = 'light',
}) {
  const canvasRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    const isDark = theme === 'dark';
    const textColor = isDark ? '#94A3B8' : '#64748B';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';

    // Build chronological labels and datasets
    const histDates = history.map((h) => h.date);
    const fcstDates = forecast.map((f) => f.date);
    const allLabels = [...histDates, ...fcstDates];

    // Data arrays aligned with allLabels
    const histActual = [
      ...history.map((h) => h.actual),
      ...Array(fcstDates.length).fill(null),
    ];

    // To connect history with forecast smoothly, seed first forecast element at last history
    const lastHistVal = history.length > 0 ? history[history.length - 1].actual : null;
    const padding = Array(Math.max(0, history.length - 1)).fill(null);

    const fcstP10 = [
      ...padding,
      lastHistVal,
      ...forecast.map((f) => f.p10),
    ];
    const fcstP50 = [
      ...padding,
      lastHistVal,
      ...forecast.map((f) => f.p50),
    ];
    const fcstP90 = [
      ...padding,
      lastHistVal,
      ...forecast.map((f) => f.p90),
    ];

    const ctx = canvasRef.current.getContext('2d');

    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'Historical Actual Sales',
            data: histActual,
            borderColor: isDark ? '#E2E8F0' : '#1E293B',
            backgroundColor: isDark ? '#E2E8F0' : '#1E293B',
            borderWidth: 2,
            pointRadius: (ctx) => (history.length > 40 ? 0 : 2.5),
            tension: 0.15,
            order: 1,
          },
          {
            label: 'P90 (Upper Bound)',
            data: fcstP90,
            borderColor: 'rgba(13, 148, 136, 0.65)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
            tension: 0.2,
            order: 2,
          },
          {
            label: 'P10–P90 Uncertainty Band',
            data: fcstP10,
            borderColor: 'rgba(13, 148, 136, 0.65)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: '-1', // fill area between P90 and P10
            backgroundColor: isDark ? 'rgba(13, 148, 136, 0.18)' : 'rgba(13, 148, 136, 0.12)',
            tension: 0.2,
            order: 3,
          },
          {
            label: 'P50 (Median Forecast)',
            data: fcstP50,
            borderColor: '#0D9488',
            backgroundColor: '#0D9488',
            borderWidth: 3,
            pointRadius: 2.5,
            pointHoverRadius: 5,
            tension: 0.2,
            order: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: textColor,
              boxWidth: 14,
              font: {
                family: 'Inter, sans-serif',
                size: 11,
              },
            },
          },
          tooltip: {
            backgroundColor: isDark ? '#0F172A' : '#1E293B',
            titleColor: '#F8FAFC',
            bodyColor: '#F8FAFC',
            borderColor: isDark ? '#334155' : '#475569',
            borderWidth: 1,
            padding: 10,
          },
        },
        scales: {
          x: {
            grid: {
              color: gridColor,
            },
            ticks: {
              color: textColor,
              maxTicksLimit: 12,
              font: {
                family: 'JetBrains Mono, monospace',
                size: 10,
              },
            },
          },
          y: {
            grid: {
              color: gridColor,
            },
            ticks: {
              color: textColor,
              font: {
                family: 'JetBrains Mono, monospace',
                size: 10,
              },
            },
            title: {
              display: true,
              text: 'Demand (Units / Sales)',
              color: textColor,
              font: {
                family: 'Inter, sans-serif',
                size: 11,
              },
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
  }, [history, forecast, theme]);

  return (
    <div style={{ position: 'relative', width: '100%', height }}>
      <canvas ref={canvasRef} />
    </div>
  );
}

