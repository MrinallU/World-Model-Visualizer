import planck, { Vec2 } from "planck-js";

// Discrete actions like Gym: 0=do nothing, 1=left, 2=main, 3=right
// Obs is [x, y, vx, vy, angle, angularVel, leftContact, rightContact]

function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

export class LunarLanderPhysics {
  // --- tunables ---
  dt = 1 / 60;
  gravity = -10;

  // “arena” scaling (roughly)
  xScale = 20;
  yScale = 20;

  // thruster forces (tune to match feel)
  mainForce = 25;
  sideForce = 6;

  // termination
  maxSteps = 1000;

  // --- state ---
  world = null;
  ground = null;
  lander = null;
  leftLeg = null;
  rightLeg = null;

  leftContact = 0;
  rightContact = 0;

  steps = 0;
  prevShaping = null;

  constructor(seed) {
    // seed optional — if you want deterministic RNG later
    void seed;
  }

  reset() {
    this.steps = 0;
    this.prevShaping = null;
    this.leftContact = 0;
    this.rightContact = 0;

    this.world = new planck.World(Vec2(0, this.gravity));

    // Ground: flat for now (you can add helipad edges later)
    this.ground = this.world.createBody();
    this.ground.createFixture(planck.Edge(Vec2(-40, 0), Vec2(40, 0)), { friction: 0.8 });

    // Lander body
    this.lander = this.world.createDynamicBody({
      position: Vec2(0, 12),
      angle: 0,
      angularDamping: 1.5,
      linearDamping: 0.2,
    });
    this.lander.createFixture(planck.Box(0.6, 0.9), {
      density: 1.0,
      friction: 0.3,
      restitution: 0.0,
    });

    // Sensor legs (for contact flags)
    this.leftLeg = this.world.createDynamicBody({
      position: this.lander.getPosition().clone().add(Vec2(-0.55, -1.1)),
    });
    this.leftLeg.createFixture(planck.Box(0.08, 0.5), { density: 0.2, isSensor: true });

    this.rightLeg = this.world.createDynamicBody({
      position: this.lander.getPosition().clone().add(Vec2(0.55, -1.1)),
    });
    this.rightLeg.createFixture(planck.Box(0.08, 0.5), { density: 0.2, isSensor: true });

    // weld to lander (cheap approximation; you can do revolute joints later)
    this.world.createJoint(planck.WeldJoint({}, this.lander, this.leftLeg, this.leftLeg.getPosition()));
    this.world.createJoint(planck.WeldJoint({}, this.lander, this.rightLeg, this.rightLeg.getPosition()));

    // contacts
    this.world.on("begin-contact", (c) => {
      const a = c.getFixtureA().getBody();
      const b = c.getFixtureB().getBody();
      if ((a === this.leftLeg && b === this.ground) || (b === this.leftLeg && a === this.ground)) this.leftContact = 1;
      if ((a === this.rightLeg && b === this.ground) || (b === this.rightLeg && a === this.ground)) this.rightContact = 1;
    });

    this.world.on("end-contact", (c) => {
      const a = c.getFixtureA().getBody();
      const b = c.getFixtureB().getBody();
      if ((a === this.leftLeg && b === this.ground) || (b === this.leftLeg && a === this.ground)) this.leftContact = 0;
      if ((a === this.rightLeg && b === this.ground) || (b === this.rightLeg && a === this.ground)) this.rightContact = 0;
    });

    return this.getObservation();
  }

  step(action) {
    this.steps++;

    this.applyAction(action);
    this.world.step(this.dt);

    const obs = this.getObservation();
    const { reward, done, info } = this.computeRewardDone(obs, action);

    return { obs, reward, done, info };
  }

  // --- observation: [x, y, vx, vy, angle, angularVel, leftContact, rightContact]
  getObservation() {
    const p = this.lander.getPosition();
    const v = this.lander.getLinearVelocity();
    const a = this.wrapAngle(this.lander.getAngle());
    const w = this.lander.getAngularVelocity();

    const x = clamp(p.x / this.xScale, -1, 1);
    const y = clamp(p.y / this.yScale, -1, 1);
    const vx = clamp(v.x / 10, -1, 1);
    const vy = clamp(v.y / 10, -1, 1);
    const ang = clamp(a, -Math.PI, Math.PI);
    const angVel = clamp(w / 5, -1, 1);

    return [x, y, vx, vy, ang, angVel, this.leftContact, this.rightContact];
  }

  computeRewardDone(obs, action) {
    const [x, y, vx, vy, ang, angVel, lC, rC] = obs;

    const dist = Math.sqrt(x * x + y * y);
    const speed = Math.sqrt(vx * vx + vy * vy);

    const shaping =
      -100 * dist -
      40 * speed -
      10 * Math.abs(ang) -
      10 * Math.abs(angVel) +
      10 * (lC + rC);

    let reward = 0;
    if (this.prevShaping !== null) reward += shaping - this.prevShaping;
    this.prevShaping = shaping;

    // fuel cost
    if (action === 2) reward -= 0.3;
    if (action === 1 || action === 3) reward -= 0.03;

    let done = false;
    const p = this.lander.getPosition();
    const crash = p.y < 0.2 && (lC + rC) === 0;
    const landed = p.y < 0.6 && (lC + rC) === 2 && speed < 0.2 && Math.abs(ang) < 0.2;

    if (Math.abs(p.x) > 40 || p.y > 50) {
      done = true;
      reward -= 100;
    }

    if (crash) {
      done = true;
      reward -= 100;
    } else if (landed) {
      done = true;
      reward += 100;
    }

    if (this.steps >= this.maxSteps) done = true;

    return { reward, done, info: { landed, crash, steps: this.steps } };
  }

  applyAction(action) {
    if (action === 0) return;

    const pos = this.lander.getWorldCenter();
    const angle = this.lander.getAngle();

    const up = Vec2(Math.sin(angle), Math.cos(angle));
    const rt = Vec2(Math.cos(angle), -Math.sin(angle));

    if (action === 2) {
      const F = up.clone().mul(this.mainForce);
      this.lander.applyForce(F, pos, true);
    } else if (action === 1) {
      const point = pos.clone().add(rt.clone().mul(0.6));
      const F = rt.clone().mul(-this.sideForce);
      this.lander.applyForce(F, point, true);
    } else if (action === 3) {
      const point = pos.clone().add(rt.clone().mul(-0.6));
      const F = rt.clone().mul(this.sideForce);
      this.lander.applyForce(F, point, true);
    }
  }

  wrapAngle(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
  }
}

