/**
 * AI Demand Forecasting and Inventory Decision Support
 * Single Source of Truth Mock Data Store
 * SRS Reference: AI_Demand_Forecasting_SRS_v3_short.md
 */

(function () {
  'use strict';

  // Helper to generate calendar dates YYYY-MM-DD
  function getDateStr(daysAgo) {
    const d = new Date(2026, 10, 7); // Base date: Nov 7, 2026
    d.setDate(d.getDate() - daysAgo);
    return d.toISOString().split('T')[0];
  }

  function getFutureDateStr(daysAhead) {
    const d = new Date(2026, 10, 7);
    d.setDate(d.getDate() + daysAhead);
    return d.toISOString().split('T')[0];
  }

  // Generate 90 days history and 28 days future forecasts for Item 17 @ Store 3
  // D_T over T=10 days is exactly 120 (average 12 units/day)
  function generateItem17Series() {
    const history = [];
    const forecast = [];

    // 90 days history (day -89 to day 0)
    for (let i = 89; i >= 0; i--) {
      const date = getDateStr(i);
      const dayOfWeek = new Date(date).getDay();
      // Weekday pattern: weekend slightly higher, base ~11.8, noise
      const weekendBoost = (dayOfWeek === 0 || dayOfWeek === 6) ? 3.5 : 0;
      const seasonal = Math.sin((89 - i) / 7 * Math.PI) * 1.8;
      const noise = ((i * 13) % 7) - 3;
      const actual = Math.max(2, Math.round(11.5 + weekendBoost + seasonal + noise));
      history.push({ date, actual, open: 1, promo: (i % 14 < 5) ? 1 : 0 });
    }

    // 28 days future forecast (day 1 to day 28)
    // First 10 days sum of P50 is constrained to sum to exactly 120
    const tenDayTargets = [11, 13, 10, 11, 14, 15, 10, 11, 12, 13]; // Sum = 120
    for (let i = 1; i <= 28; i++) {
      const date = getFutureDateStr(i);
      const dayOfWeek = new Date(date).getDay();
      let p50;
      if (i <= 10) {
        p50 = tenDayTargets[i - 1];
      } else {
        const weekendBoost = (dayOfWeek === 0 || dayOfWeek === 6) ? 3.5 : 0;
        p50 = Math.round(11.8 + weekendBoost + (((i * 7) % 5) - 2));
      }
      // Uncertainty expands over horizon
      const horizonFactor = 1 + (i / 35);
      const p10 = Math.max(1, Math.round(p50 - (3.8 * horizonFactor)));
      const p90 = Math.round(p50 + (4.6 * horizonFactor));
      forecast.push({ date, p10, p50, p90 });
    }

    return { history, forecast };
  }

  const item17Data = generateItem17Series();

  const MOCK_DATA = {
    // Current Active Dataset ("store-item" or "rossmann")
    activeDataset: 'store-item',

    // Disclaimers required by SRS Section 2 & Section 6
    disclaimer: "Stock and lead-time values are simulated/user-entered.",
    originDisclaimer: "Datasets are Kaggle benchmarks and do not represent Pakistani retail.",

    // Store Item Demand Dataset (Store x Item)
    storeItemSeries: [
      {
        id: 'item-17-store-3',
        item: 17,
        store: 3,
        name: 'Organic Rolled Oats 1kg',
        category: 'Dry Grocery',
        stock: 40,             // S
        incoming: 0,           // I
        leadTime: 3,           // L (days)
        reviewPeriod: 7,       // R (days)
        serviceLevelZ: 1.65,   // z (~95%)
        sigmaD: 5.7,           // Daily validation RMSE
        d_T: 120,              // Forecast demand over T=10 days
        p90LeadTimeDemand: 48, // P90 demand over L=3 days (15+16+17 = 48)
        history: item17Data.history,
        forecast: item17Data.forecast,
        unitCost: 4.20,
        unitPrice: 6.99
      },
      {
        id: 'item-12-store-1',
        item: 12,
        store: 1,
        name: 'Whole Roasted Coffee Beans 500g',
        category: 'Beverages',
        stock: 15,
        incoming: 0,
        leadTime: 5,
        reviewPeriod: 7,
        serviceLevelZ: 1.65,
        sigmaD: 7.2,
        d_T: 165,
        p90LeadTimeDemand: 82, // S + I (15) < 82 -> Stockout Risk!
        history: [],
        forecast: [],
        unitCost: 8.50,
        unitPrice: 13.99
      },
      {
        id: 'item-44-store-5',
        item: 44,
        store: 5,
        name: 'Greek Yogurt 1kg',
        category: 'Dairy',
        stock: 85,
        incoming: 50,
        leadTime: 4,
        reviewPeriod: 10,
        serviceLevelZ: 1.65,
        sigmaD: 6.8,
        d_T: 210,
        p90LeadTimeDemand: 95, // S + I (135) > 95, but Q > 0 -> Reorder recommended
        history: [],
        forecast: [],
        unitCost: 3.10,
        unitPrice: 5.49
      },
      {
        id: 'item-3-store-2',
        item: 3,
        store: 2,
        name: 'Extra Virgin Olive Oil 750ml',
        category: 'Oils & Vinegars',
        stock: 220,
        incoming: 40,
        leadTime: 3,
        reviewPeriod: 7,
        serviceLevelZ: 1.65,
        sigmaD: 4.1,
        d_T: 95,
        p90LeadTimeDemand: 42, // S + I (260) > D_T + SS -> Healthy, Q = 0
        history: [],
        forecast: [],
        unitCost: 9.00,
        unitPrice: 15.20
      },
      {
        id: 'item-8-store-3',
        item: 8,
        store: 3,
        name: 'Almond Butter Crunchy 350g',
        category: 'Spreads',
        stock: 50,
        incoming: 20,
        leadTime: 4,
        reviewPeriod: 7,
        serviceLevelZ: 1.65,
        sigmaD: 4.9,
        d_T: 110,
        p90LeadTimeDemand: 52, // S+I (70) > 52 -> Reorder recommended
        history: [],
        forecast: [],
        unitCost: 5.80,
        unitPrice: 9.50
      },
      {
        id: 'item-19-store-7',
        item: 19,
        store: 7,
        name: 'Basmati Rice Aged 5kg',
        category: 'Grains',
        stock: 12,
        incoming: 10,
        leadTime: 6,
        reviewPeriod: 7,
        serviceLevelZ: 1.96,   // 97.5%
        sigmaD: 8.4,
        d_T: 190,
        p90LeadTimeDemand: 98, // S + I (22) < 98 -> Severe Stockout Risk!
        history: [],
        forecast: [],
        unitCost: 12.00,
        unitPrice: 18.50
      },
      {
        id: 'item-28-store-4',
        item: 28,
        store: 4,
        name: 'Raw Wildflower Honey 500g',
        category: 'Sweeteners',
        stock: 310,
        incoming: 0,
        leadTime: 2,
        reviewPeriod: 5,
        serviceLevelZ: 1.65,
        sigmaD: 3.2,
        d_T: 65,
        p90LeadTimeDemand: 28, // Healthy, Q = 0
        history: [],
        forecast: [],
        unitCost: 6.20,
        unitPrice: 10.99
      },
      {
        id: 'item-35-store-6',
        item: 35,
        store: 6,
        name: 'Sparkling Mineral Water 1L x 6',
        category: 'Beverages',
        stock: 190,
        incoming: 60,
        leadTime: 3,
        reviewPeriod: 7,
        serviceLevelZ: 1.65,
        sigmaD: 5.0,
        d_T: 130,
        p90LeadTimeDemand: 55, // Healthy, Q = 0
        history: [],
        forecast: [],
        unitCost: 4.00,
        unitPrice: 7.20
      }
    ],

    // Rossmann Store Sales Dataset (Store-Level Daily Sales)
    rossmannSeries: [
      {
        id: 'rossmann-store-1',
        store: 1,
        storeType: 'c',
        assortment: 'a',
        competitionDistance: 1270,
        name: 'Store 1 — Central High St',
        stock: 850,
        incoming: 400,
        leadTime: 3,
        reviewPeriod: 7,
        serviceLevelZ: 1.65,
        sigmaD: 340,
        d_T: 52000,
        p90LeadTimeDemand: 18500,
        promoActive: 1,
        closedSundays: true,
        history: [],
        forecast: []
      },
      {
        id: 'rossmann-store-3',
        store: 3,
        storeType: 'a',
        assortment: 'a',
        competitionDistance: 14130,
        name: 'Store 3 — Regional Shopping Park',
        stock: 320,
        incoming: 200,
        leadTime: 4,
        reviewPeriod: 7,
        serviceLevelZ: 1.96,
        sigmaD: 580,
        d_T: 78000,
        p90LeadTimeDemand: 34000,
        promoActive: 0,
        closedSundays: true,
        history: [],
        forecast: []
      },
      {
        id: 'rossmann-store-25',
        store: 25,
        storeType: 'a',
        assortment: 'c',
        competitionDistance: 430,
        name: 'Store 25 — Downtown Station Metro',
        stock: 1200,
        incoming: 800,
        leadTime: 2,
        reviewPeriod: 5,
        serviceLevelZ: 1.65,
        sigmaD: 420,
        d_T: 64000,
        p90LeadTimeDemand: 22000,
        promoActive: 1,
        closedSundays: false, // Open 7 days
        history: [],
        forecast: []
      }
    ],

    // Models Registry (FR-13)
    models: [
      {
        id: 'v1.2.0-lgb',
        version: 'v1.2.0',
        name: 'LightGBM Multi-Quantile (P10/P50/P90)',
        architecture: 'Gradient Boosted Trees (Quantile Regression Objective)',
        trainedDate: '2026-10-15',
        validationWindow: '2026-08-01 to 2026-09-30 (Expanding Window)',
        rmse: 5.70,
        mae: 4.12,
        wmape: 11.4,
        coverage: 80.4,
        status: 'active',
        isCurrent: true,
        notes: 'Current production model. P10-P90 coverage calibrated to 80.4% across 50 items.'
      },
      {
        id: 'v1.3.0-candidate',
        version: 'v1.3.0',
        name: 'LightGBM + Lag Interactions (Candidate)',
        architecture: 'LightGBM with 7d/14d/28d rolling quantile features',
        trainedDate: '2026-11-01',
        validationWindow: '2026-09-01 to 2026-10-31 (Expanding Window)',
        rmse: 5.58, // Better than current (5.70)
        mae: 4.02,
        wmape: 11.1,
        coverage: 80.8,
        status: 'validated',
        isCurrent: false,
        canSwitch: true,
        notes: 'Retrained in Colab with updated October data. Performance is improved across all horizons.'
      },
      {
        id: 'v1.1.0-lgb',
        version: 'v1.1.0',
        name: 'LightGBM Baseline Quantile',
        architecture: 'LightGBM with basic lag features',
        trainedDate: '2026-09-20',
        validationWindow: '2026-07-01 to 2026-08-31',
        rmse: 5.88,
        mae: 4.28,
        wmape: 11.9,
        coverage: 79.5,
        status: 'archived',
        isCurrent: false,
        canSwitch: false,
        notes: 'Superseded by v1.2.0.'
      },
      {
        id: 'v0.9.0-degraded',
        version: 'v0.9.0-exp',
        name: 'LightGBM Deep Trees (Overfitted Experiment)',
        architecture: 'LightGBM with max_depth=16, learning_rate=0.08',
        trainedDate: '2026-09-10',
        validationWindow: '2026-07-01 to 2026-08-31',
        rmse: 6.42, // Worse than current (5.70)
        mae: 4.88,
        wmape: 13.5,
        coverage: 74.2,
        status: 'rejected',
        isCurrent: false,
        canSwitch: false,
        notes: 'Degraded metrics on holdout test window. Activation blocked per SRS Section 5/FR-13.'
      },
      {
        id: 'v0.5.0-seasonal-naive',
        version: 'v0.5.0-sn',
        name: 'Seasonal Naive (7-day lag benchmark)',
        architecture: 'Lag 7 heuristic baseline',
        trainedDate: '2026-09-01',
        validationWindow: 'Full holdout comparison',
        rmse: 7.82,
        mae: 5.94,
        wmape: 16.5,
        coverage: 68.2,
        status: 'benchmark',
        isCurrent: false,
        canSwitch: false,
        notes: 'Reference benchmark model for academic evaluation.'
      },
      {
        id: 'v0.3.0-moving-avg',
        version: 'v0.3.0-ma',
        name: '14-Day Rolling Moving Average',
        architecture: 'Simple rolling window average',
        trainedDate: '2026-09-01',
        validationWindow: 'Full holdout comparison',
        rmse: 8.45,
        mae: 6.51,
        wmape: 18.1,
        coverage: 64.5,
        status: 'benchmark',
        isCurrent: false,
        canSwitch: false,
        notes: 'Reference naive baseline.'
      }
    ],

    // Evaluation Comparison (FR-11)
    evaluation: {
      modelComparison: [
        {
          model: 'LightGBM (Quantile)',
          type: 'ML Model (Multi-horizon)',
          mae: 4.12,
          rmse: 5.70,
          wmape: 11.4,
          mase: 0.69,
          coverageP10P90: 80.4,
          trainingTime: '2.4 min (Colab T4)',
          inferenceLatency: '18 ms/series'
        },
        {
          model: 'Seasonal Naive (7-day lag)',
          type: 'Deterministic Baseline',
          mae: 5.94,
          rmse: 7.82,
          wmape: 16.5,
          mase: 1.00,
          coverageP10P90: 68.2,
          trainingTime: '< 1 sec',
          inferenceLatency: '2 ms/series'
        },
        {
          model: '14-Day Moving Average',
          type: 'Deterministic Baseline',
          mae: 6.51,
          rmse: 8.45,
          wmape: 18.1,
          mase: 1.10,
          coverageP10P90: 64.5,
          trainingTime: '< 1 sec',
          inferenceLatency: '1 ms/series'
        }
      ],
      // Error by Forecast Horizon (Days 1 to 14)
      horizonError: [
        { day: 'Day 1', lgbMAE: 3.20, snMAE: 5.15, maMAE: 5.80, lgbCoverage: 81.2 },
        { day: 'Day 2', lgbMAE: 3.35, snMAE: 5.22, maMAE: 5.90, lgbCoverage: 80.9 },
        { day: 'Day 3', lgbMAE: 3.52, snMAE: 5.30, maMAE: 6.05, lgbCoverage: 80.7 },
        { day: 'Day 4', lgbMAE: 3.71, snMAE: 5.48, maMAE: 6.18, lgbCoverage: 80.5 },
        { day: 'Day 5', lgbMAE: 3.88, snMAE: 5.62, maMAE: 6.30, lgbCoverage: 80.4 },
        { day: 'Day 6', lgbMAE: 4.05, snMAE: 5.80, maMAE: 6.42, lgbCoverage: 80.2 },
        { day: 'Day 7', lgbMAE: 4.15, snMAE: 5.94, maMAE: 6.51, lgbCoverage: 80.4 },
        { day: 'Day 8', lgbMAE: 4.28, snMAE: 6.08, maMAE: 6.65, lgbCoverage: 80.1 },
        { day: 'Day 9', lgbMAE: 4.41, snMAE: 6.19, maMAE: 6.78, lgbCoverage: 79.9 },
        { day: 'Day 10', lgbMAE: 4.56, snMAE: 6.28, maMAE: 6.90, lgbCoverage: 79.8 },
        { day: 'Day 11', lgbMAE: 4.68, snMAE: 6.35, maMAE: 7.01, lgbCoverage: 79.6 },
        { day: 'Day 12', lgbMAE: 4.79, snMAE: 6.42, maMAE: 7.12, lgbCoverage: 79.5 },
        { day: 'Day 13', lgbMAE: 4.88, snMAE: 6.48, maMAE: 7.20, lgbCoverage: 79.4 },
        { day: 'Day 14', lgbMAE: 4.95, snMAE: 6.54, maMAE: 7.28, lgbCoverage: 79.2 }
      ],
      // Quantile Calibration Details
      calibration: [
        { quantile: 'P10', nominal: 10.0, empirical: 9.8, difference: -0.2, status: 'Calibrated' },
        { quantile: 'P50', nominal: 50.0, empirical: 50.3, difference: +0.3, status: 'Calibrated' },
        { quantile: 'P90', nominal: 90.0, empirical: 90.2, difference: +0.2, status: 'Calibrated' },
        { quantile: 'P10–P90 Band', nominal: 80.0, empirical: 80.4, difference: +0.4, status: 'Ideal' }
      ]
    },

    // Manager Decision Log (FR-12)
    decisions: [
      {
        id: 'dec-101',
        timestamp: '2026-11-06 14:22',
        series: 'Item 12 @ Store 1',
        action: 'Accept',
        recommendedQ: 185,
        finalQ: 185,
        reason: 'Severe stockout risk detected; accepted full recommended order.',
        user: 'Fatima N. (Inventory Lead)'
      },
      {
        id: 'dec-102',
        timestamp: '2026-11-06 11:05',
        series: 'Item 44 @ Store 5',
        action: 'Adjust',
        recommendedQ: 140,
        finalQ: 100,
        reason: 'Adjusted down due to warehouse cold-storage shelf limits.',
        user: 'Zaid A. (Supply Ops)'
      },
      {
        id: 'dec-103',
        timestamp: '2026-11-05 16:45',
        series: 'Item 8 @ Store 3',
        action: 'Reject',
        recommendedQ: 95,
        finalQ: 0,
        reason: 'Supplier announced promotional packaging redesign next week.',
        user: 'Fatima N. (Inventory Lead)'
      }
    ]
  };

  // Expose to global scope for client consumption
  window.MOCK_DATA = MOCK_DATA;
})();

