"""犯规检测模块：垫步、踩线、多人入界、出界、撑杆辅助等规则。"""


class FoulDetector:
    def __init__(self, calibrator, kpt_idx, get_kpt, get_feet, transform_to_mat_cm):
        self.calibrator = calibrator
        self.kpt_idx = kpt_idx
        self.get_kpt = get_kpt
        self.get_feet = get_feet
        self.transform_to_mat_cm = transform_to_mat_cm
        self.reason = None

    def reset(self):
        self.reason = None

    def check_step_jump(self, front_toe_hist, takeoff_toe_x_cm):
        if not front_toe_hist or takeoff_toe_x_cm is None:
            return
        initial_x = front_toe_hist[0][0]
        diff = takeoff_toe_x_cm - initial_x
        if diff > 10.0:
            self._set_reason("垫步 (Step Jump)", f"检测到犯规动作: 垫步 (Step Jump), diff={diff:.1f}cm")

    def check_single_leg_takeoff(self, kpts):
        if kpts is None:
            return
        left_ankle = self.get_kpt(kpts, self.kpt_idx["l_ankle"])
        right_ankle = self.get_kpt(kpts, self.kpt_idx["r_ankle"])
        if left_ankle is None or right_ankle is None:
            return
        diff_x = abs(left_ankle[0] - right_ankle[0])
        diff_y = abs(left_ankle[1] - right_ankle[1])
        threshold_px = 30.0
        if self.calibrator.px_per_cm > 0:
            threshold_px = 15.0 * self.calibrator.px_per_cm
        if diff_x > threshold_px or diff_y > threshold_px:
            self._set_reason(
                "垫步 (Step Jump)",
                f"检测到犯规动作: 垫步 (Step Jump) (单脚起跳特征), dx={diff_x:.1f}, dy={diff_y:.1f}",
            )

    def check_multi_person(self, all_kpts_list):
        if not self.calibrator.calibrated or len(all_kpts_list) < 2:
            return
        valid_people_in_mat = 0
        for person_kpts in all_kpts_list:
            ankles = self.get_feet(person_kpts, "ankle")
            toes = self.get_feet(person_kpts, "toe")
            heels = self.get_feet(person_kpts, "heel")
            points = []
            for feet in [ankles, toes, heels]:
                if feet["l"]:
                    points.append(feet["l"])
                if feet["r"]:
                    points.append(feet["r"])
            if any(self.calibrator.strict_in_mat(self.transform_to_mat_cm(pt)) for pt in points):
                valid_people_in_mat += 1
        if valid_people_in_mat >= 2:
            self._set_reason("多人入界 (Multi-Person)", f"检测到犯规动作: 多人入界 (Multi-Person), count={valid_people_in_mat}")

    def check_prop_assistance(self, kpts):
        if kpts is None:
            return
        lw = self.get_kpt(kpts, self.kpt_idx["l_wrist"])
        rw = self.get_kpt(kpts, self.kpt_idx["r_wrist"])
        lk = self.get_kpt(kpts, self.kpt_idx["l_knee"])
        rk = self.get_kpt(kpts, self.kpt_idx["r_knee"])
        if lw is None or rw is None or lk is None or rk is None:
            return
        if lw[1] > lk[1] or rw[1] > rk[1]:
            self._set_reason("撑杆/异物 (Prop Assistance)", "检测到犯规动作: 撑杆/异物 (Prop Assistance)")

    def check_out_of_bounds(self, landing_xy_cm):
        if landing_xy_cm is None:
            return
        _, y_cm = landing_xy_cm
        if y_cm < 0.0 or y_cm > self.calibrator.mat_width_cm:
            self._set_reason("出界 (Out of Bounds)", "检测到犯规动作: 出界 (Out of Bounds)")

    def check_line_violation(self, current_toe_x_cm, takeoff_line_cm):
        if current_toe_x_cm is None:
            return
        if current_toe_x_cm > (takeoff_line_cm + 1.0):
            self._set_reason("踩线 (Line Violation)", "检测到犯规动作: 踩线 (Line Violation)")

    def _set_reason(self, reason, message):
        if self.reason != reason:
            print(message)
        self.reason = reason
