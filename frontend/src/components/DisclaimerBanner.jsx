import React from 'react';

export default function DisclaimerBanner() {
  return (
    <aside className="disclaimer-banner" role="complementary" aria-label="System Disclaimers">
      <div className="mandatory-notice">
        <span className="badge-sim">SIMULATION NOTICE</span>
        <span>Stock and lead-time values are simulated/user-entered.</span>
      </div>
      <div className="origin-notice">
        Benchmark datasets: Rossmann Store Sales &amp; Store Item Demand (do not represent Pakistani retail).
      </div>
    </aside>
  );
}

