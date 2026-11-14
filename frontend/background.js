/**
 * AI Demand Forecasting and Inventory Decision Support
 * 3D Background: "The Probability Ribbon"
 * 
 * References & Credits:
 * - https://github.com/ruucm/shadergradient (3D animated noise gradient meshes)
 * - https://github.com/pmndrs/react-three-fiber (Scene composition & render pipeline patterns)
 */

(function () {
  'use strict';

  // Check URL params for lite mode (?lite=1)
  const urlParams = new URLSearchParams(window.location.search);
  const isLiteMode = urlParams.get('lite') === '1';
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Fallback handler if WebGL is unavailable or lite mode active
  if (isLiteMode || prefersReducedMotion) {
    console.log('[Background] Lite or reduced-motion mode active: Using optimized static background.');
    const canvas = document.getElementById('bg-canvas');
    if (canvas) canvas.style.display = 'none';
    document.body.classList.add('bg-static-fallback');
    window.bgRibbon = {
      setMood: () => {},
      pulse: () => {},
      setTheme: () => {}
    };
    return;
  }

  // Ensure Three.js is loaded
  if (typeof THREE === 'undefined') {
    console.warn('[Background] Three.js not found. Falling back to static gradient.');
    document.body.classList.add('bg-static-fallback');
    return;
  }

  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;

  // Scene setup
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 10);

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  } catch (err) {
    console.warn('[Background] WebGL initialization failed. Falling back to static gradient.', err);
    document.body.classList.add('bg-static-fallback');
    return;
  }

  // Custom GLSL Shaders for the Probability Ribbon
  // Vertex Shader: Simplex Noise Displacement that fans outward along the timeline (P10-P90 cone)
  const vertexShader = `
    uniform float uTime;
    uniform float uPulse;
    uniform float uMood;
    varying vec2 vUv;
    varying float vElevation;
    varying vec3 vNormal;

    // Classical 3D Perlin / Simplex Noise helper
    vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
    vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

    float snoise(vec3 v){
      const vec2 C = vec2(1.0/6.0, 1.0/3.0);
      const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
      vec3 i  = floor(v + dot(v, C.yyy));
      vec3 x0 = v - i + dot(i, C.xxx);
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min(g.xyz, l.zxy);
      vec3 i2 = max(g.xyz, l.zxy);
      vec3 x1 = x0 - i1 + 1.0 * C.xxx;
      vec3 x2 = x0 - i2 + 2.0 * C.xxx;
      vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
      i = mod(i, 289.0);
      vec4 p = permute(permute(permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0));
      float n_ = 0.142857142857;
      vec3  ns = n_ * D.wyz - D.xzx;
      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_);
      vec4 x = x_ *ns.x + ns.yyyy;
      vec4 y = y_ *ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);
      vec4 b0 = vec4(x.xy, y.xy);
      vec4 b1 = vec4(x.zw, y.zw);
      vec4 s0 = floor(b0)*2.0 + 1.0;
      vec4 s1 = floor(b1)*2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));
      vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
      vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
      vec3 p0 = vec3(a0.xy, h.x);
      vec3 p1 = vec3(a0.zw, h.y);
      vec3 p2 = vec3(a1.xy, h.z);
      vec3 p3 = vec3(a1.zw, h.w);
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
      p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }

    void main() {
      vUv = uv;
      vNormal = normal;

      // Speed increases slightly when in alert mood
      float speed = uTime * (0.15 + uMood * 0.1);
      
      // Uncertainty widening: displacement increases towards the right (future horizon)
      float horizonFactor = smoothstep(0.0, 1.0, uv.x);
      float dispersion = 0.5 + horizonFactor * 1.5;

      // Multi-frequency noise
      float noise1 = snoise(vec3(position.x * 0.25, position.y * 0.35 + speed, speed * 0.5));
      float noise2 = snoise(vec3(position.x * 0.6, position.y * 0.7 - speed * 0.7, uTime * 0.2)) * 0.5;
      
      float elevation = (noise1 + noise2) * dispersion * (1.0 + uPulse * 0.8);
      
      // Arc shape along X to make it cradle the viewport gracefully
      elevation += sin(uv.x * 3.14159) * 0.8;

      vElevation = elevation;

      vec3 newPosition = position;
      newPosition.z += elevation;

      gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
    }
  `;

  // Fragment Shader: Smooth gradient coloring with median P50 glow and translucent P10/P90 interval
  const fragmentShader = `
    uniform float uTime;
    uniform float uMood;
    uniform float uTheme; // 0.0 = dark, 1.0 = light
    varying vec2 vUv;
    varying float vElevation;
    varying vec3 vNormal;

    void main() {
      // Color Palettes
      // Dark Base: deep navy ink #0B1120
      vec3 darkBase = vec3(0.043, 0.067, 0.125);
      // Dark Teal median: #0D9488
      vec3 darkTeal = vec3(0.051, 0.580, 0.533);
      // Light Cyan accent: #2DD4BF
      vec3 lightCyan = vec3(0.176, 0.831, 0.749);
      // Alert warm amber/red: #DC2626 & #D97706
      vec3 alertRed = vec3(0.863, 0.149, 0.149);
      vec3 alertAmber = vec3(0.851, 0.467, 0.024);

      // Light mode colors (soft, translucent, pearlescent)
      vec3 lightBase = vec3(0.965, 0.975, 0.985);
      vec3 lightTeal = vec3(0.80, 0.94, 0.92);

      // Interpolate median flow vs mood
      vec3 primaryTeal = mix(darkTeal, lightCyan, vElevation * 0.5 + 0.5);
      vec3 alertColor = mix(alertAmber, alertRed, clamp(vElevation * 0.8 + 0.2, 0.0, 1.0));
      vec3 activeColor = mix(primaryTeal, alertColor, uMood);

      // Central median line highlight
      float medianLine = 1.0 - smoothstep(0.0, 0.25, abs(vUv.y - 0.5));
      
      // Uncertainty interval envelope (fades towards edges)
      float edgeFade = smoothstep(0.0, 0.2, vUv.y) * (1.0 - smoothstep(0.8, 1.0, vUv.y));

      vec3 finalDark = mix(darkBase, activeColor, edgeFade * 0.45);
      finalDark += primaryTeal * medianLine * 0.35;

      vec3 finalLight = mix(lightBase, lightTeal, edgeFade * 0.5);
      finalLight += darkTeal * medianLine * 0.15;

      vec3 col = mix(finalDark, finalLight, uTheme);

      // Soft vignette transparency at outer edges
      float alpha = mix(0.75, 0.45, uTheme) * edgeFade;

      gl_FragColor = vec4(col, alpha);
    }
  `;

  // Create Uniforms
  const isInitialDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const uniforms = {
    uTime: { value: 0.0 },
    uMood: { value: 0.3 }, // 0.0 calm, 1.0 alert
    uPulse: { value: 0.0 },
    uTheme: { value: isInitialDark ? 0.0 : 1.0 }
  };

  // Ribbon Mesh Geometry
  const geometry = new THREE.PlaneGeometry(26, 15, 120, 70);
  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide
  });

  const ribbonMesh = new THREE.Mesh(geometry, material);
  ribbonMesh.rotation.x = -Math.PI * 0.22;
  ribbonMesh.rotation.z = -0.05;
  ribbonMesh.position.set(1.5, -0.5, -3.5);
  scene.add(ribbonMesh);

  // Floating Micro-Particles ("Inventory Flow Units")
  const particleCount = 120;
  const particleGeo = new THREE.BufferGeometry();
  const particlePositions = new Float32Array(particleCount * 3);
  const particleSpeeds = new Float32Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    particlePositions[i * 3] = (Math.random() - 0.5) * 22;
    particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 8;
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 6 - 2;
    particleSpeeds[i] = 0.005 + Math.random() * 0.015;
  }

  particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));

  const particleMat = new THREE.PointsMaterial({
    size: 0.065,
    color: new THREE.Color(0x2dd4bf),
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending
  });

  const particleSystem = new THREE.Points(particleGeo, particleMat);
  scene.add(particleSystem);

  // Mouse Parallax
  let mouseX = 0;
  let mouseY = 0;
  let targetCameraX = 0;
  let targetCameraY = 0;

  window.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    targetCameraX = mouseX * 0.6;
    targetCameraY = mouseY * 0.4;
  }, { passive: true });

  // Resize Handler
  function handleResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  }
  window.addEventListener('resize', handleResize, { passive: true });

  // Animation Loop with Visibility Optimization
  let isPageVisible = !document.hidden;
  document.addEventListener('visibilitychange', () => {
    isPageVisible = !document.hidden;
  });

  let clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    if (!isPageVisible) return;

    const delta = clock.getDelta();
    const elapsedTime = clock.getElapsedTime();

    uniforms.uTime.value = elapsedTime;

    // Decay pulse smoothly
    if (uniforms.uPulse.value > 0.001) {
      uniforms.uPulse.value *= 0.94;
    } else {
      uniforms.uPulse.value = 0;
    }

    // Camera parallax lerp
    camera.position.x += (targetCameraX - camera.position.x) * 0.04;
    camera.position.y += (targetCameraY - camera.position.y) * 0.04;
    camera.lookAt(0, 0, 0);

    // Animate particles flowing across horizon
    const positions = particleGeo.attributes.position.array;
    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] += particleSpeeds[i] * (1.0 + uniforms.uMood.value * 0.5);
      // Recycle particle when it exits view
      if (positions[i * 3] > 11) {
        positions[i * 3] = -11;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 8;
      }
    }
    particleGeo.attributes.position.needsUpdate = true;

    renderer.render(scene, camera);
  }

  animate();

  // Public Controller API
  window.bgRibbon = {
    setMood: (mode) => {
      // 'alert' -> 0.85 (tenser, warmer tint for stockout alerts)
      // 'calm' -> 0.0 (clean teal/blue for evaluation and models)
      // 'normal' -> 0.35 (default dashboard)
      const targetMood = mode === 'alert' ? 0.85 : mode === 'calm' ? 0.0 : 0.35;
      gsapLikeLerp(uniforms.uMood, targetMood, 800);
    },
    pulse: () => {
      uniforms.uPulse.value = 1.0;
    },
    setTheme: (theme) => {
      const targetVal = theme === 'dark' ? 0.0 : 1.0;
      uniforms.uTheme.value = targetVal;
      particleMat.color.setHex(theme === 'dark' ? 0x2dd4bf : 0x0d9488);
      particleMat.opacity = theme === 'dark' ? 0.55 : 0.25;
    }
  };

  // Helper for smooth uniform lerping
  function gsapLikeLerp(uniformObj, target, durationMs) {
    const start = uniformObj.value;
    const startTime = performance.now();
    function step(now) {
      const progress = Math.min((now - startTime) / durationMs, 1.0);
      const ease = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      uniformObj.value = start + (target - start) * ease;
      if (progress < 1.0) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

})();

