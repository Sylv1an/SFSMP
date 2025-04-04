# flight_sim.py
import pygame
import math
import sys
import random
import time

from rocket_data import RocketBlueprint, PlacedPart
from parts import draw_part_shape, get_part_data
from ui_elements import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRAY, RED, GREEN, BLUE

# --- Flight Sim Constants ---
GRAVITY = 9.81 * 8; ROTATION_SPEED = 200; COLLISION_DAMAGE_FACTOR = 0.6
MIN_IMPACT_VEL_DAMAGE = 1.5; THROTTLE_CHANGE_RATE = 0.5
REACTION_WHEEL_TORQUE = 15000; ANGULAR_DAMPING = 0.3
GROUND_Y = 1000; WORLD_WIDTH = 5000; BLUE_SKY_Y_LIMIT = -2000; SPACE_Y_LIMIT = -15000
STAR_COUNT = 200; STAR_FIELD_DEPTH = 10000
COLOR_SKY_BLUE = pygame.Color(135, 206, 250); COLOR_SPACE_BLACK = pygame.Color(0, 0, 0)
COLOR_HORIZON = pygame.Color(170, 210, 230); COLOR_GROUND = pygame.Color(0, 150, 0)
COLOR_FLAME = pygame.Color(255, 100, 0); COLOR_UI_BAR = pygame.Color(0, 200, 0)
COLOR_UI_BAR_BG = pygame.Color(50, 50, 50)
COLOR_EXPLOSION = [pygame.Color(255,255,0), pygame.Color(255,150,0), pygame.Color(200,50,0), pygame.Color(GRAY)]
COLOR_ENGINE_ENABLED = GREEN; COLOR_ENGINE_DISABLED = RED

# --- Simple Particle Class for Explosions ---
class ExplosionEffect:
    def __init__(self, pos, num_particles=15, max_life=0.5, max_speed=100):
        self.particles=[]; self.pos=pygame.math.Vector2(pos); self.max_life=max_life
        for _ in range(num_particles): a=random.uniform(0,360); s=random.uniform(max_speed*0.2,max_speed); v=pygame.math.Vector2(s,0).rotate(a); l=random.uniform(max_life*0.3,max_life); c=random.choice(COLOR_EXPLOSION); r=random.uniform(2,5); self.particles.append([self.pos.copy(),v,l,l,c,r]); self.is_alive=True
    def update(self, dt):
        if not self.is_alive: return
        apf = False
        for p in self.particles:
            if p[2]>0: p[0]+=p[1]*dt; p[1]*=0.95; p[2]-=dt; apf=True
        self.is_alive = apf
    def draw(self, surface, camera):
        if not self.is_alive: return
        for p in self.particles:
            rl = p[2];
            if rl > 0: sp=camera.apply(p[0]); ml=p[3]; af=max(0,rl/ml)if ml>0 else 0; bc=p[4]; br=p[5]; dc=(int(bc.r*af),int(bc.g*af),int(bc.b*af)); dr=int(br*af);
            if dc!=(0,0,0) and dr > 0: pygame.draw.circle(surface,dc,sp,dr)

# --- Camera Class ---
class Camera:
    def __init__(self, width, height): self.camera_rect=pygame.Rect(0,0,width,height); self.width=width; self.height=height; self.offset=pygame.math.Vector2(0,0)
    def apply(self, target_pos): return target_pos-self.offset
    def apply_rect(self, target_rect): return target_rect.move(-self.offset.x,-self.offset.y)
    def update(self, target_pos): x=target_pos.x-self.width//2; y=target_pos.y-self.height//2; self.offset=pygame.math.Vector2(x,y)

# --- FlyingRocket Class ---
class FlyingRocket:
    def __init__(self, blueprint: RocketBlueprint, initial_pos, initial_angle=0):
        self.blueprint_name = blueprint.name; self.parts = [PlacedPart.from_dict(p.to_dict()) for p in blueprint.parts]
        if not self.parts: raise ValueError("Cannot initialize FlyingRocket with no parts.")
        self.original_root_part_obj = self.parts[0]; self.pos = pygame.math.Vector2(initial_pos)
        self.vel = pygame.math.Vector2(0,0); self.acc = pygame.math.Vector2(0,0); self.angle = initial_angle; self.angular_velocity = 0.0
        self.thrusting = False; self.landed = False; self.effects = []; self.throttle_level = 0.0; self.master_thrust_enabled = False
        self.engines = []; self.fuel_tanks = []; total_fuel_cap = 0; total_dry_mass = 0
        self.total_mass = 0; self.moment_of_inertia = 10000; self.center_of_mass_offset = pygame.math.Vector2(0,0)
        for i, part in enumerate(self.parts):
            part.current_hp = part.part_data.get("max_hp",100); part.is_broken = False; part.relative_angle = 0; part.part_index = i; part.engine_enabled = True
            pt = part.part_data.get("type")
            if pt=="Engine": self.engines.append(part)
            elif pt=="FuelTank": self.fuel_tanks.append(part)
            total_fuel_cap += part.part_data.get("fuel_capacity",0); total_dry_mass += part.part_data.get("mass",0)
        self.current_fuel = total_fuel_cap; self.dry_mass = total_dry_mass; self.fuel_mass_per_unit = 0.1
        self.calculate_physics_properties(); self.calculate_bounds()

    def calculate_physics_properties(self):
        total_m = 0; com_numerator = pygame.math.Vector2(0, 0); fuel_mass_total = self.current_fuel * self.fuel_mass_per_unit
        if not self.parts: self.total_mass = 0.01; self.center_of_mass_offset = pygame.math.Vector2(0,0); self.moment_of_inertia = 1.0; return
        total_tank_capacity = sum(p.part_data.get("fuel_capacity", 0) for p in self.fuel_tanks); total_tank_capacity = max(1, total_tank_capacity)
        for part in self.parts:
            part_mass = part.part_data.get("mass", 0)
            if part.part_data.get("type") == "FuelTank": part_mass += fuel_mass_total * (part.part_data.get("fuel_capacity", 0) / total_tank_capacity)
            total_m += part_mass; com_numerator += part.relative_pos * part_mass
        self.total_mass = max(0.01, total_m)
        self.center_of_mass_offset = com_numerator / self.total_mass if self.total_mass > 0.01 else pygame.math.Vector2(0, 0)
        self.calculate_bounds(); height = self.local_bounds.height if self.local_bounds else 10
        self.moment_of_inertia = max(1.0, (1/12.0) * self.total_mass * (height ** 2))

    def calculate_bounds(self):
        x0,x1,y0,y1=0,0,0,0;
        if not self.parts:self.local_bounds=pygame.Rect(0,0,0,0);return
        for p in self.parts:hw=p.part_data['width']/2;hh=p.part_data['height']/2;px=p.relative_pos.x;py=p.relative_pos.y;x0=min(x0,px-hw);x1=max(x1,px+hw);y0=min(y0,py-hh);y1=max(y1,py+hh)
        self.local_bounds=pygame.Rect(x0,y0,x1-x0,y1-y0)
    def get_parts_near_world_pos(self, wp:pygame.math.Vector2, r:float=15.0):
        nb=[];rsq=r*r; [p for p in self.parts if (self.pos+p.relative_pos.rotate(-self.angle)-wp).length_squared()<rsq and nb.append(p)]; return nb
    def get_lowest_point_world(self) -> pygame.math.Vector2:
        ly=-float('inf'); lpw=self.pos; found=False
        if not self.parts: return self.pos
        for p in self.parts: w=p.part_data['width']; h=p.part_data['height']; prp=p.relative_pos.rotate(-self.angle); pcw=self.pos+prp; pta=self.angle+p.relative_angle; cs=[pygame.math.Vector2(x,h/2) for x in [-w/2,w/2,0]];
        for c in cs: rco=c.rotate(-pta); cw=pcw+rco;
        if cw.y>ly: ly=cw.y; lpw=cw; found=True
        if not found: bco=pygame.math.Vector2(self.local_bounds.centerx, self.local_bounds.bottom); return self.pos+bco.rotate(-self.angle)
        return lpw
    def update_total_mass(self): pass
    def apply_force(self, force_vector):
        if self.total_mass <= 0.01: return
        self.acc += force_vector / self.total_mass
    def consume_fuel(self, a):
        c=min(self.current_fuel,a);
        if c > 0: self.current_fuel-=c; self.calculate_physics_properties(); return True
        return False
    def get_thrust_data(self) -> tuple[pygame.math.Vector2, pygame.math.Vector2 | None, float]:
        ttfv = pygame.math.Vector2(0,0); wap = None; tc_rate = 0; aec = 0; ttm = 0; tpn = pygame.math.Vector2(0,0)
        if not self.master_thrust_enabled or self.throttle_level<=0 or self.current_fuel<=0: return ttfv, wap, tc_rate
        for ep in self.engines:
            if ep.engine_enabled:
                thrust_mag = ep.part_data.get("thrust",0); cons = ep.part_data.get("fuel_consumption",0)
                ttm += thrust_mag; tc_rate += cons; aec += 1; tpn += ep.relative_pos * thrust_mag
        if aec>0 and ttm>0:
            current_thrust_magnitude = ttm * self.throttle_level
            atr = tpn / ttm if ttm > 0 else pygame.math.Vector2(0,0)
            rad_angle = math.radians(self.angle); thrust_direction = pygame.math.Vector2(-math.sin(rad_angle), -math.cos(rad_angle))
            ttfv = thrust_direction * current_thrust_magnitude
            rtr = atr.rotate(-self.angle); wap = self.pos+rtr
        return ttfv, wap, tc_rate
    def apply_collision_damage(self, ivm):
        if ivm<MIN_IMPACT_VEL_DAMAGE: return
        dmg=(ivm**1.5)*COLLISION_DAMAGE_FACTOR; print(f"Impact:{ivm:.1f} Dmg:{dmg:.1f}"); lp=self.get_lowest_point_world(); ip=self.get_parts_near_world_pos(lp); ptd=[]
        if not ip and self.parts: lpo=min(self.parts,key=lambda p:(p.relative_pos.rotate(-self.angle)+self.pos).y,default=None); ip=[lpo] if lpo else []
        for p in ip:
            if p.is_broken: continue
            p.current_hp-=dmg; print(f" {p.part_id} HP:{p.current_hp:.0f}/{p.part_data.get('max_hp',1)}");
            if p.current_hp<=0: print(f" > {p.part_id} BROKEN!"); p.is_broken=True; p.current_hp=0; ptd.append(p); prp=p.relative_pos.rotate(-self.angle); pcw=self.pos+prp; self.effects.append(ExplosionEffect(pcw))
        if ptd: self.destroy_parts(ptd)
    def destroy_parts(self, ptd):
        if not ptd: return
        print(f"Destroying:{[p.part_id for p in ptd]}"); self.parts=[p for p in self.parts if p not in ptd]; self.engines=[e for e in self.engines if e not in ptd]; self.fuel_tanks=[t for t in self.fuel_tanks if t not in ptd];
        self.calculate_physics_properties(); self.calculate_bounds();
        if not self.parts: print("All parts gone!")

    # *** CORRECTED toggle_engine_at_pos method ***
    def toggle_engine_at_pos(self, click_world_pos):
        # --- ADD CHECK: Return immediately if no engines exist ---
        if not self.engines:
            return False

        click_radius_sq = 15**2
        toggled = False
        # Loop through existing engines
        for engine in self.engines:
            rotated_rel_pos = engine.relative_pos.rotate(-self.angle)
            engine_center_world = self.pos + rotated_rel_pos # ecw is calculated here
            dist_sq = (engine_center_world - click_world_pos).length_squared()
            # Check if click is close enough
            if dist_sq < click_radius_sq: # Check happens here
                engine.engine_enabled = not engine.engine_enabled
                print(f"Toggled engine {engine.part_id} {'ON' if engine.engine_enabled else 'OFF'}")
                toggled = True
                break # Important: exit loop after finding one engine
        return toggled

    def update(self, dt):
        if not self.parts: self.pos+=self.vel*dt; self.vel*=0.99; self.acc=pygame.math.Vector2(0,0); [e.update(dt) for e in self.effects]; self.effects=[e for e in self.effects if e.is_alive]; return
        angular_vel_change = 0.0; net_force = pygame.math.Vector2(0, 0); self.acc = pygame.math.Vector2(0,0)
        gravity_force = pygame.math.Vector2(0, GRAVITY * self.total_mass); self.apply_force(gravity_force)
        thrust_force_potential, thrust_app_point_world, consumption_rate_100 = self.get_thrust_data()
        if thrust_force_potential.length_squared() > 0:
            fuel_needed = consumption_rate_100 * self.throttle_level * dt
            if self.consume_fuel(fuel_needed):
                self.thrusting = True; self.apply_force(thrust_force_potential)
                if thrust_app_point_world:
                    com_offset_rotated = self.center_of_mass_offset.rotate(-self.angle); world_com = self.pos + com_offset_rotated
                    r_thrust = thrust_app_point_world - world_com; torque_thrust = r_thrust.x * thrust_force_potential.y - r_thrust.y * thrust_force_potential.x
                    if self.moment_of_inertia > 0: angular_vel_change += math.degrees(torque_thrust / self.moment_of_inertia) * dt
            else: self.thrusting = False
        else: self.thrusting = False
        can_control = self.original_root_part_obj in self.parts and not self.original_root_part_obj.is_broken
        if can_control:
            keys = pygame.key.get_pressed(); control_torque_input = 0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: control_torque_input += REACTION_WHEEL_TORQUE
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: control_torque_input -= REACTION_WHEEL_TORQUE
            if self.moment_of_inertia > 0 and control_torque_input != 0: angular_vel_change += math.degrees(control_torque_input / self.moment_of_inertia) * dt
        self.vel += self.acc * dt; self.pos += self.vel * dt
        self.angular_velocity += angular_vel_change; self.angular_velocity *= (1.0 - ANGULAR_DAMPING * dt)
        self.angle = (self.angle + self.angular_velocity * dt) % 360
        self.landed = False; vb4 = self.vel.copy()
        lp = self.get_lowest_point_world()
        if lp.y >= GROUND_Y:
            self.landed = True; ivm = vb4.length(); self.apply_collision_damage(ivm)
            if self.parts: lp_ad=self.get_lowest_point_world(); c=lp_ad.y-GROUND_Y; self.pos.y-=c; self.vel.y=0; self.vel.x*=0.8; self.angular_velocity = 0
            else: self.vel=pygame.math.Vector2(0,0); self.landed=False
        [e.update(dt) for e in self.effects]; self.effects = [e for e in self.effects if e.is_alive]

    def draw(self, s, cam):
        nbv=0
        for p in self.parts:
            prp=p.relative_pos.rotate(-self.angle); pcw=self.pos+prp; psp=cam.apply(pcw); pta=self.angle+p.relative_angle; ic=None
            if p.part_data.get("type")=="Engine": ic=COLOR_ENGINE_ENABLED if p.engine_enabled else COLOR_ENGINE_DISABLED
            draw_part_shape(s,p.part_data,psp,pta,broken=p.is_broken);
            if ic: pygame.draw.circle(s,ic,psp,4)
            if p.is_broken: nbv+=1
        if self.thrusting:
            fs = 0.5 + 0.5 * self.throttle_level
            for ep in self.engines:
                if ep.engine_enabled:
                    prp=ep.relative_pos.rotate(-self.angle); ecw=self.pos+prp; eta=self.angle+ep.relative_angle
                    fbo=pygame.math.Vector2(0,ep.part_data["height"]/2).rotate(-eta); fbw=ecw+fbo
                    fl=25*fs; fw=ep.part_data["width"]*0.9*fs
                    fdw=pygame.math.Vector2(0,1).rotate(-eta); fsw=pygame.math.Vector2(1,0).rotate(-eta)
                    ftw=fbw+fdw*fl; flw=fbw-fsw*fw/2; frw=fbw+fsw*fw/2
                    fps=[cam.apply(p) for p in [flw,frw,ftw]]; pygame.draw.polygon(s,COLOR_FLAME,fps)
        [e.draw(s,cam) for e in self.effects]; return nbv

# --- Background/Terrain Functions ---
def create_stars(c,b):
    s=[]
    for _ in range(c):
        x = random.uniform(b.left, b.right); y = random.uniform(b.top, b.bottom)
        z = random.uniform(1, STAR_FIELD_DEPTH); s.append((pygame.math.Vector2(x, y), z))
    return s
def draw_earth_background(s, cam, stars):
    sr = s.get_rect(); avy = cam.offset.y + cam.height / 2
    if avy > BLUE_SKY_Y_LIMIT:
        hys = cam.apply(pygame.math.Vector2(0, GROUND_Y)).y
        if hys < sr.bottom: pygame.draw.rect(s, COLOR_HORIZON, (0, hys, sr.width, sr.height - hys)); pygame.draw.rect(s, COLOR_SKY_BLUE, (0, 0, sr.width, hys))
        else: s.fill(COLOR_SKY_BLUE)
    elif avy < SPACE_Y_LIMIT:
        s.fill(COLOR_SPACE_BLACK); draw_stars(s, stars, cam, alpha=255)
    else:
        i = max(0., min(1., (avy - BLUE_SKY_Y_LIMIT) / (SPACE_Y_LIMIT - BLUE_SKY_Y_LIMIT)))
        s.fill(COLOR_SKY_BLUE.lerp(COLOR_SPACE_BLACK, i)); sa = int(255 * i)
        if sa > 10: draw_stars(s, stars, cam, alpha=sa)
def draw_stars(s, stars, cam, alpha=255):
    cw=cam.offset+pygame.math.Vector2(SCREEN_WIDTH/2,SCREEN_HEIGHT/2); bsc=pygame.Color(GRAY)
    for spw, z in stars:
        pf = 1. / (z / 100 + 1); svp = cw + (spw - cw) * pf; ssp = cam.apply(svp)
        if 0<=ssp.x<SCREEN_WIDTH and 0<=ssp.y<SCREEN_HEIGHT:
            sz=max(1,int(2*(1.-z/STAR_FIELD_DEPTH))); af=alpha/255.; fc=(int(bsc.r*af),int(bsc.g*af),int(bsc.b*af))
            if fc!=(0,0,0): pygame.draw.circle(s,fc,(int(ssp.x),int(ssp.y)),sz)
def draw_terrain(s,cam):
    gvr=pygame.Rect(cam.offset.x-50,GROUND_Y,cam.width+100,SCREEN_HEIGHT*2); grs=cam.apply_rect(gvr); pygame.draw.rect(s,COLOR_GROUND,grs)

# --- Simulation Runner Function ---
def run_simulation(screen, clock, blueprint_file):
    print(f"Attempting to load blueprint: {blueprint_file}")
    initial_blueprint = RocketBlueprint.load_from_json(blueprint_file)
    if not initial_blueprint or not initial_blueprint.parts: print("BP load failed/empty."); return
    start_x = 0; lowest_offset_y = initial_blueprint.get_lowest_point_offset_y(); start_y = GROUND_Y - lowest_offset_y
    initial_pos = (start_x, start_y)
    print(f"Lowest point offset relative to root: {lowest_offset_y:.2f}"); print(f"Calculated Start Y: {start_y:.2f} (Ground Y: {GROUND_Y})")
    try:
        player_rocket = FlyingRocket(initial_blueprint, initial_pos)
    except Exception as e: print(f"Rocket init error: {e}"); return

    camera = Camera(SCREEN_WIDTH, SCREEN_HEIGHT); camera.update(player_rocket.pos)
    star_area = pygame.Rect(-WORLD_WIDTH*2, SPACE_Y_LIMIT-5000, WORLD_WIDTH*4, abs(SPACE_Y_LIMIT)+GROUND_Y+10000)
    stars = create_stars(STAR_COUNT*2, star_area); ui_font = pygame.font.SysFont(None, 24)

    sim_running = True
    while sim_running:
        dt = clock.tick(60)/1000.0; dt = min(dt, 0.1)

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: sim_running = False
                if event.key == pygame.K_SPACE: player_rocket.master_thrust_enabled = not player_rocket.master_thrust_enabled; print(f"Master Thrust: {'ON' if player_rocket.master_thrust_enabled else 'OFF'}")
                if event.key == pygame.K_r:
                    print("RESPAWNING ROCKET...")
                    try: player_rocket = FlyingRocket(initial_blueprint, initial_pos); camera.update(player_rocket.pos); print("Rocket Respawned.")
                    except Exception as e: print(f"Respawn Error: {e}"); sim_running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                 click_screen_pos = pygame.math.Vector2(event.pos); click_world_pos = click_screen_pos + camera.offset
                 player_rocket.toggle_engine_at_pos(click_world_pos)

        # Continuous Input (Throttle)
        keys = pygame.key.get_pressed(); throttle_change = 0
        if keys[pygame.K_w] or keys[pygame.K_UP]: throttle_change += THROTTLE_CHANGE_RATE * dt
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: throttle_change -= THROTTLE_CHANGE_RATE * dt
        if throttle_change != 0: player_rocket.throttle_level = max(0.0, min(1.0, player_rocket.throttle_level + throttle_change))

        # Updates
        player_rocket.update(dt); camera.update(player_rocket.pos)

        # Drawing
        draw_earth_background(screen, camera, stars); draw_terrain(screen, camera)
        num_broken_visually = player_rocket.draw(screen, camera)

        # UI Overlay
        bar_width=20; bar_height=100; bar_x=15; bar_y=SCREEN_HEIGHT-bar_height-30
        pygame.draw.rect(screen,COLOR_UI_BAR_BG,(bar_x,bar_y,bar_width,bar_height)); fill_height=bar_height*player_rocket.throttle_level
        pygame.draw.rect(screen,COLOR_UI_BAR,(bar_x,bar_y+bar_height-fill_height,bar_width,fill_height)); pygame.draw.rect(screen,WHITE,(bar_x,bar_y,bar_width,bar_height),1)
        th_txt=ui_font.render("Thr",True,WHITE);screen.blit(th_txt,(bar_x,bar_y+bar_height+5))

        if player_rocket.parts:
            alt_agl=max(0,GROUND_Y-player_rocket.get_lowest_point_world().y)
            has_control=player_rocket.original_root_part_obj in player_rocket.parts and not player_rocket.original_root_part_obj.is_broken
            cs="OK" if has_control else "NO CONTROL"; mts="ON" if player_rocket.master_thrust_enabled else "OFF"
            st=[f"Thr:{player_rocket.throttle_level*100:.0f}% ({mts})",f"Fuel:{player_rocket.current_fuel:.1f}",
                f"VelX:{player_rocket.vel.x:.1f}",f"VelY:{player_rocket.vel.y:.1f}",f"Alt:{alt_agl:.1f}",
                f"AngVel:{player_rocket.angular_velocity:.1f} d/s",
                f"Parts:{len(player_rocket.parts)}({num_broken_visually})",f"Ctrl:{cs}"]
            if player_rocket.original_root_part_obj in player_rocket.parts: rp=player_rocket.original_root_part_obj; st.append(f"ROOT HP:{rp.current_hp:.0f}/{rp.part_data.get('max_hp',1)}")
            t_y=10; tc=WHITE if has_control else RED
            for i,t in enumerate(st): ts=ui_font.render(t,True,tc); screen.blit(ts,(bar_x+bar_width+10,t_y+i*20))
        else: dt_txt=ui_font.render("DESTROYED",True,RED); tr=dt_txt.get_rect(center=(SCREEN_WIDTH//2,SCREEN_HEIGHT//2)); screen.blit(dt_txt,tr)

        pygame.display.flip()
    print("Exiting simulation.")