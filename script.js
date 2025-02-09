<!-- Simulation and UI Script -->
<script>
  /******************************************************
   * Global Simulation, RL, and Rendering Variables
   ******************************************************/
  // Simulation mode and constant actions.
  let simulationType = "1-pole";
  const numActions = 2;  // Two actions: left (-1) and right (+1)
  
  // Physics parameters.
  let gravity = 9.8;
  const massCart = 1.0;
  const massPole = 0.1;
  const totalMassSingle = massCart + massPole;
  const totalMassDouble = massCart + 2 * massPole;
  let poleLength = 0.5;    // (meters; half-length)
  let forceMag = 10.0;
  const baseTau = 0.02;    // simulation time step (seconds)
  let simulationSpeed = 1.0;  // speed multiplier
  
  // Cart and pole state variables.
  let x = 0, x_dot = 0;
  // For 1-pole:
  let theta = 0.05, theta_dot = 0;
  // For 2-pole:
  let theta1 = 0.05, theta1_dot = 0;
  let theta2 = 0.05, theta2_dot = 0;
  
  // Rendering conversion factors.
  const scaleFactor = 100;  // meters-to-pixels conversion.
  const offsetY = 100;      // vertical offset in pixels.
  let trajectory = [];      // stores the pole tip positions.
  
  // RL data storage.
  let states = [];
  let actions = [];
  let rewards = [];
  let episodeReward = 0, episodeCount = 0;
  
  // 2D Canvas context.
  let canvas, ctx;
  
  // TensorFlow.js models and optimizer.
  let actorModel, criticModel, optimizer;
  
  // Epsilon for epsilon‑greedy exploration.
  let epsilon = 1.0;
  const epsilonDecay = 0.995;
  const minEpsilon = 0.1;
  
  /******************************************************
   * Canvas Initialization
   ******************************************************/
  function initCanvas() {
    // Now we use the id "glcanvas" (matching the HTML)
    canvas = document.getElementById("glcanvas");
    ctx = canvas.getContext("2d");
  }
  
  /******************************************************
   * Model Setup (Actor–Critic)
   ******************************************************/
  function setupActorCriticModels() {
    const stateDim = simulationType === "1-pole" ? 4 : 6;
    // Dispose previous models if needed.
    if (actorModel) actorModel.dispose();
    if (criticModel) criticModel.dispose();
    
    // Define actor model.
    actorModel = tf.sequential();
    actorModel.add(tf.layers.dense({ inputShape: [stateDim], units: 24, activation: 'relu' }));
    actorModel.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    actorModel.add(tf.layers.dense({ units: numActions, activation: 'softmax' }));
    
    // Define critic model.
    criticModel = tf.sequential();
    criticModel.add(tf.layers.dense({ inputShape: [stateDim], units: 24, activation: 'relu' }));
    criticModel.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    criticModel.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    
    optimizer = tf.train.adam(0.01);
  }
  
  setupActorCriticModels();
  
  /******************************************************
   * Action Selection: Epsilon‑Greedy
   ******************************************************/
  function chooseAction(state) {
    if (Math.random() < epsilon) {
      return Math.random() < 0.5 ? -1 : 1;
    } else {
      return tf.tidy(() => {
        const logits = actorModel.predict(tf.tensor2d([state]));
        const probs = logits.dataSync();
        return (Math.random() < probs[0]) ? -1 : 1;
      });
    }
  }
  
  /******************************************************
   * Simulation Step Functions
   ******************************************************/
  // 1‑pole dynamics.
  function simulateSingleStep(action, dt) {
    const force = action * forceMag;
    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);
    const temp = (force + massPole * poleLength * theta_dot * theta_dot * sintheta) / totalMassSingle;
    const theta_acc = (gravity * sintheta - costheta * temp) /
                      (poleLength * (4.0/3.0 - (massPole * costheta * costheta) / totalMassSingle));
    const x_acc = temp - (massPole * poleLength * theta_acc * costheta) / totalMassSingle;
    x += dt * x_dot;
    x_dot += dt * x_acc;
    theta += dt * theta_dot;
    theta_dot += dt * theta_acc;
  }
  
  // 2‑pole dynamics (simplified/demonstrative).
  function simulateDoubleStep(action, dt) {
    const force = action * forceMag;
    let m_eff = 2 * massPole;
    let costheta1 = Math.cos(theta1);
    let sintheta1 = Math.sin(theta1);
    let temp = (force + m_eff * poleLength * theta1_dot * theta1_dot * sintheta1) / (massCart + m_eff);
    let theta1_acc = (gravity * sintheta1 - costheta1 * temp) /
                     (poleLength * (4.0/3.0 - (m_eff * costheta1 * costheta1) / (massCart + m_eff)));
    let x_acc = temp - (m_eff * poleLength * theta1_acc * costheta1) / (massCart + m_eff);
    // Simplified dynamics for the second pole.
    let theta2_acc = (gravity * Math.sin(theta2) - 0.2 * theta2_dot);
    
    x += dt * x_dot;
    x_dot += dt * x_acc;
    theta1 += dt * theta1_dot;
    theta1_dot += dt * theta1_acc;
    theta2 += dt * theta2_dot;
    theta2_dot += dt * theta2_acc;
  }
  
  /******************************************************
   * Training Function (Actor–Critic)
   ******************************************************/
  function trainActorCriticModel() {
    const discountedReturns = [];
    let sum = 0;
    const gamma = 0.99;
    for (let i = rewards.length - 1; i >= 0; i--) {
      sum = rewards[i] + gamma * sum;
      discountedReturns[i] = sum;
    }
    
    const statesTensor = tf.tensor2d(states);
    const actionsTensor = tf.tensor1d(actions, 'int32');
    const returnsTensor = tf.tensor1d(discountedReturns);
    
    optimizer.minimize(() => {
      // Actor forward pass.
      const logits = actorModel.predict(statesTensor);
      const actionProbs = tf.softmax(logits);
      
      // Critic forward pass.
      const values = criticModel.predict(statesTensor).reshape([statesTensor.shape[0]]);
      
      const advantages = returnsTensor.sub(values);
      
      const oneHotActions = tf.oneHot(actionsTensor, numActions);
      const logProbs = tf.log(tf.clipByValue(actionProbs, 1e-7, 1));
      const selectedLogProbs = tf.mul(oneHotActions, logProbs).sum(1);
      const actorLoss = selectedLogProbs.mul(advantages).neg().mean();
      
      const entropy = tf.mul(actionProbs, logProbs).sum(1).neg().mean();
      const entBonusWeight = 0.01;
      const entropyBonus = entropy.mul(entBonusWeight);
      
      const criticLoss = advantages.square().mean();
      
      const totalLoss = actorLoss.add(criticLoss).sub(entropyBonus);
      return totalLoss;
    },
    actorModel.trainableWeights.concat(criticModel.trainableWeights));
    
    statesTensor.dispose();
    actionsTensor.dispose();
    returnsTensor.dispose();
  }
  
  /******************************************************
   * 2D Canvas Rendering Functions
   ******************************************************/
  function renderScene() {
    // Clear the canvas.
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#e6e6e6";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw ground.
    const groundY = canvas.height / 2 + offsetY;
    ctx.strokeStyle = "rgb(102, 102, 102)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(canvas.width, groundY);
    ctx.stroke();
    
    // Determine cart position.
    const cartX = canvas.width / 2 + x * scaleFactor;
    const cartY = canvas.height / 2 + offsetY;
    
    // Draw cart.
    ctx.fillStyle = "black";
    ctx.fillRect(cartX - 25, cartY - 10, 50, 20);
    
    const L = poleLength * scaleFactor;
    
    if (simulationType === "1-pole") {
      const tipX = cartX + Math.sin(theta) * L;
      const tipY = cartY - Math.cos(theta) * L;
      ctx.strokeStyle = "red";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(cartX, cartY);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();
      trajectory.push({ x: tipX, y: tipY });
    } else if (simulationType === "2-pole") {
      const tip1X = cartX + Math.sin(theta1) * L;
      const tip1Y = cartY - Math.cos(theta1) * L;
      ctx.strokeStyle = "red";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(cartX, cartY);
      ctx.lineTo(tip1X, tip1Y);
      ctx.stroke();
      const tip2X = tip1X + Math.sin(theta1 + theta2) * L;
      const tip2Y = tip1Y - Math.cos(theta1 + theta2) * L;
      ctx.strokeStyle = "green";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(tip1X, tip1Y);
      ctx.lineTo(tip2X, tip2Y);
      ctx.stroke();
      trajectory.push({ x: tip2X, y: tip2Y });
    }
    
    if (trajectory.length > 500)
      trajectory.shift();
    
    if (trajectory.length > 1) {
      ctx.strokeStyle = "rgba(0, 0, 255, 0.6)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(trajectory[0].x, trajectory[0].y);
      for (let i = 1; i < trajectory.length; i++) {
        ctx.lineTo(trajectory[i].x, trajectory[i].y);
      }
      ctx.stroke();
    }
  }
  
  /******************************************************
   * Failure and Reset Functions
   ******************************************************/
  function checkFailure() {
    const outOfBounds = (Math.abs(x * scaleFactor) > canvas.width / 2);
    if (simulationType === "1-pole") {
      return (Math.abs(theta) > Math.PI / 2 || outOfBounds);
    } else if (simulationType === "2-pole") {
      return (Math.abs(theta1) > Math.PI / 2 ||
              Math.abs(theta1 + theta2) > Math.PI / 2 ||
              outOfBounds);
    }
  }
  
  function resetSimulation() {
    x = 0;
    x_dot = 0;
    if (simulationType === "1-pole") {
      theta = 0.05;
      theta_dot = 0;
    } else if (simulationType === "2-pole") {
      theta1 = 0.05;
      theta1_dot = 0;
      theta2 = 0.05;
      theta2_dot = 0;
    }
    trajectory = [];
  }
  
  /******************************************************
   * UI and Recording Setup
   ******************************************************/
  let simRunning = true;
  let mediaRecorder, recordedChunks = [];
  let recording = false;
  
  const toggleSimBtn = document.getElementById("toggleSim");
  const resetSimBtn = document.getElementById("resetSim");
  const toggleRecordBtn = document.getElementById("toggleRecord");
  const forceSlider = document.getElementById("forceSlider");
  const forceDisplay = document.getElementById("forceDisplay");
  const gravitySlider = document.getElementById("gravitySlider");
  const gravityDisplay = document.getElementById("gravityDisplay");
  const speedSlider = document.getElementById("speedSlider");
  const speedDisplay = document.getElementById("speedDisplay");
  const metricsDiv = document.getElementById("metrics");
  const poleConfigSelect = document.getElementById("poleConfig");
  
  toggleSimBtn.addEventListener("click", () => {
    simRunning = !simRunning;
    toggleSimBtn.textContent = simRunning ? "Pause Simulation" : "Resume Simulation";
  });
  
  resetSimBtn.addEventListener("click", resetSimulation);
  
  forceSlider.addEventListener("input", () => {
    forceMag = parseFloat(forceSlider.value);
    forceDisplay.textContent = forceMag.toFixed(1);
  });
  
  gravitySlider.addEventListener("input", () => {
    gravity = parseFloat(gravitySlider.value);
    gravityDisplay.textContent = gravity.toFixed(1);
  });
  
  speedSlider.addEventListener("input", () => {
    simulationSpeed = parseFloat(speedSlider.value);
    speedDisplay.textContent = simulationSpeed.toFixed(1);
  });
  
  poleConfigSelect.addEventListener("change", () => {
    simulationType = poleConfigSelect.value;
    resetSimulation();
    states = [];
    actions = [];
    rewards = [];
    episodeReward = 0;
    setupActorCriticModels();
  });
  
  toggleRecordBtn.addEventListener("click", () => {
    if (!recording) {
      startRecording();
      toggleRecordBtn.textContent = "Stop Recording";
    } else {
      stopRecording();
      toggleRecordBtn.textContent = "Start Recording";
    }
    recording = !recording;
  });
  
  function startRecording() {
    let stream = canvas.captureStream(30);
    mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = saveRecording;
    mediaRecorder.start();
  }
  
  function stopRecording() { mediaRecorder.stop(); }
  
  function saveRecording() {
    let blob = new Blob(recordedChunks, { type: "video/webm" });
    recordedChunks = [];
    let url = URL.createObjectURL(blob);
    let a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = "simulation.webm";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  }
  
  function updateMetrics() {
    if (simulationType === "1-pole") {
      metricsDiv.innerHTML = `
        Episode: ${episodeCount}<br>
        Episode Reward: ${episodeReward}<br>
        x: ${x.toFixed(2)}, x_dot: ${x_dot.toFixed(2)}<br>
        theta: ${theta.toFixed(2)}, theta_dot: ${theta_dot.toFixed(2)}
      `;
    } else if (simulationType === "2-pole") {
      metricsDiv.innerHTML = `
        Episode: ${episodeCount}<br>
        Episode Reward: ${episodeReward}<br>
        x: ${x.toFixed(2)}, x_dot: ${x_dot.toFixed(2)}<br>
        theta1: ${theta1.toFixed(2)}, theta1_dot: ${theta1_dot.toFixed(2)}<br>
        theta2: ${theta2.toFixed(2)}, theta2_dot: ${theta2_dot.toFixed(2)}
      `;
    }
  }
  
  /******************************************************
   * Main Animation Loop
   ******************************************************/
  function step() {
    if (simRunning) {
      const dt = baseTau * simulationSpeed;
      let currentState, action;
      if (simulationType === "1-pole") {
        currentState = [x, x_dot, theta, theta_dot];
        action = chooseAction(currentState);
        simulateSingleStep(action, dt);
        states.push(currentState);
        actions.push(action === -1 ? 0 : 1);
        rewards.push(Math.abs(theta) < Math.PI/6 ? 1 : 0);
      } else if (simulationType === "2-pole") {
        currentState = [x, x_dot, theta1, theta1_dot, theta2, theta2_dot];
        action = chooseAction(currentState);
        simulateDoubleStep(action, dt);
        states.push(currentState);
        actions.push(action === -1 ? 0 : 1);
        rewards.push((Math.abs(theta1) < Math.PI/6 && Math.abs(theta1 + theta2) < Math.PI/6) ? 1 : 0);
      }
      episodeReward += rewards[rewards.length - 1];
      
      if (checkFailure()) {
        episodeCount++;
        console.log("Episode:", episodeCount, "Total reward:", episodeReward);
        trainActorCriticModel();
        epsilon = Math.max(minEpsilon, epsilon * epsilonDecay);
        resetSimulation();
        states = [];
        actions = [];
        rewards = [];
        episodeReward = 0;
      }
    }
    renderScene();
    updateMetrics();
    requestAnimationFrame(step);
  }
  
  /******************************************************
   * Initialization and Start
   ******************************************************/
  initCanvas();
  resetSimulation();
  requestAnimationFrame(step);
</script>
