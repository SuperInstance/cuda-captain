//! # cuda-captain
//!
//! Captain-class vessel — orchestrates fleets, manages deliberation,
//! and makes high-stakes decisions. The bridge of the ship.
//!
//! ```rust
//! use cuda_captain::{Captain, FleetCommand, Mission};
//! ```
//!
//! Captains don't execute tasks — they direct the vessels that do.


use std::collections::{HashMap, VecDeque};

/// A command from the captain to the fleet.
#[derive(Debug, Clone, PartialEq)]
pub enum FleetCommand {
    /// Assign a task to a vessel.
    Assign { vessel_id: u64, task: String, priority: u8 },
    /// Recall a vessel from its current task.
    Recall { vessel_id: u64 },
    /// Broadcast to all vessels.
    Broadcast { message: String },
    /// Form a temporary working group.
    FormGroup { name: String, members: Vec<u64>, task: String },
    /// Disband a working group.
    DisbandGroup { name: String },
    /// Set fleet-wide operating parameters.
    SetPolicy { key: String, value: String },
    /// Emergency stop.
    AllStop,
    /// Resume operations.
    Resume,
}

/// Mission status.
#[derive(Debug, Clone, PartialEq)]
pub enum MissionStatus {
    Planning,
    InProgress,
    Blocked { reason: String },
    Completed,
    Failed { reason: String },
    Abandoned,
}

/// A mission the captain is managing.
#[derive(Debug, Clone)]
pub struct Mission {
    pub id: u64,
    pub name: String,
    pub objective: String,
    pub status: MissionStatus,
    pub assigned_vessels: Vec<u64>,
    pub sub_tasks: Vec<SubTask>,
    pub confidence: f64,
    pub created_at: u64,
    pub deadline: Option<u64>,
}

impl Mission {
    pub fn new(id: u64, name: &str, objective: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self { id, name: name.to_string(), objective: objective.to_string(),
            status: MissionStatus::Planning, assigned_vessels: vec![],
            sub_tasks: vec![], confidence: 0.5,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
            deadline: None }
    }

    pub fn with_deadline(mut self, ms: u64) -> Self { self.deadline = Some(ms); self }

    pub fn add_subtask(&mut self, name: &str, assignee: u64) {
        self.sub_tasks.push(SubTask { name: name.to_string(), assignee,
            status: TaskState::Pending, result: None });
    }

    pub fn progress(&self) -> f64 {
        if self.sub_tasks.is_empty() { return 0.0; }
        let done = self.sub_tasks.iter().filter(|t| t.status == TaskState::Done).count();
        done as f64 / self.sub_tasks.len() as f64
    }

    pub fn is_overdue(&self) -> bool {
        match self.deadline {
            Some(dl) => now_ms() > dl,
            None => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskState { Pending, Assigned, InProgress, Done, Failed }

#[derive(Debug, Clone)]
pub struct SubTask {
    pub name: String,
    pub assignee: u64,
    pub status: TaskState,
    pub result: Option<String>,
}

/// The Captain — fleet orchestrator.
pub struct Captain {
    pub name: String,
    pub confidence: f64,
    vessels: HashMap<u64, VesselRecord>,
    missions: HashMap<u64, Mission>,
    command_log: VecDeque<FleetCommand>,
    next_mission_id: u64,
    policies: HashMap<String, String>,
    max_concurrent_missions: usize,
}

/// Captain's view of a vessel.
#[derive(Debug, Clone)]
pub struct VesselRecord {
    pub id: u64,
    pub name: String,
    pub trust: f64,
    pub current_task: Option<String>,
    pub last_seen: u64,
    pub health: VesselHealth,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VesselHealth { Nominal, Degraded, Critical, Offline }

impl Captain {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), confidence: 0.8,
            vessels: HashMap::new(), missions: HashMap::new(),
            command_log: VecDeque::new(), next_mission_id: 1,
            policies: HashMap::new(), max_concurrent_missions: 3 }
    }

    /// Add a vessel to the captain's fleet.
    pub fn enlist(&mut self, id: u64, name: &str, trust: f64) {
        self.vessels.insert(id, VesselRecord {
            id, name: name.to_string(), trust: trust.clamp(0.0, 1.0),
            current_task: None, last_seen: now_ms(), health: VesselHealth::Nominal });
    }

    /// Issue a command.
    pub fn command(&mut self, cmd: FleetCommand) -> CommandResult {
        self.command_log.push_back(cmd.clone());
        match cmd {
            FleetCommand::Assign { vessel_id, task, priority } => {
                if let Some(v) = self.vessels.get_mut(&vessel_id) {
                    v.current_task = Some(task.clone());
                    v.last_seen = now_ms();
                    CommandResult::Issued { target: vessel_id, detail: format!("Assigned: {}", task) }
                } else {
                    CommandResult::Failed { reason: format!("Vessel {} not found", vessel_id) }
                }
            }
            FleetCommand::Recall { vessel_id } => {
                if let Some(v) = self.vessels.get_mut(&vessel_id) {
                    let prev = v.current_task.take();
                    CommandResult::Issued { target: vessel_id, detail: format!("Recalled from: {:?}", prev) }
                } else {
                    CommandResult::Failed { reason: format!("Vessel {} not found", vessel_id) }
                }
            }
            FleetCommand::Broadcast { message } => {
                for v in self.vessels.values_mut() { v.last_seen = now_ms(); }
                CommandResult::Broadcast { recipients: self.vessels.len(), message }
            }
            FleetCommand::AllStop => {
                for v in self.vessels.values_mut() { v.current_task = None; }
                CommandResult::FleetWide { detail: "All stop ordered".into() }
            }
            FleetCommand::Resume => {
                CommandResult::FleetWide { detail: "Operations resumed".into() }
            }
            FleetCommand::SetPolicy { key, value } => {
                self.policies.insert(key, value);
                CommandResult::PolicySet
            }
            FleetCommand::FormGroup { name, members, task } => {
                CommandResult::GroupFormed { name, members: members.len(), task }
            }
            FleetCommand::DisbandGroup { name } => {
                CommandResult::GroupDisbanded { name }
            }
        }
    }

    /// Create a mission.
    pub fn plan_mission(&mut self, name: &str, objective: &str) -> Option<u64> {
        let active = self.missions.values().filter(|m| m.status == MissionStatus::InProgress).count();
        if active >= self.max_concurrent_missions { return None; }

        let id = self.next_mission_id;
        self.next_mission_id += 1;
        self.missions.insert(id, Mission::new(id, name, objective));
        Some(id)
    }

    /// Launch a mission by assigning vessels.
    pub fn launch_mission(&mut self, mission_id: u64, vessel_ids: Vec<u64>) -> bool {
        let mission = match self.missions.get_mut(&mission_id) {
            Some(m) if m.status == MissionStatus::Planning => m,
            _ => return false,
        };
        mission.assigned_vessels = vessel_ids.clone();
        mission.status = MissionStatus::InProgress;
        for vid in vessel_ids {
            if let Some(v) = self.vessels.get_mut(&vid) { v.current_task = Some(format!("mission_{}", mission_id)); }
        }
        true
    }

    /// Complete a subtask on a mission.
    pub fn complete_subtask(&mut self, mission_id: u64, subtask_name: &str, result: &str) {
        if let Some(mission) = self.missions.get_mut(&mission_id) {
            if let Some(task) = mission.sub_tasks.iter_mut().find(|t| t.name == subtask_name) {
                task.status = TaskState::Done;
                task.result = Some(result.to_string());
                mission.confidence = (mission.confidence + 0.05).min(1.0);
                // Check if mission complete
                if mission.sub_tasks.iter().all(|t| t.status == TaskState::Done) {
                    mission.status = MissionStatus::Completed;
                }
            }
        }
    }

    /// Get most trusted available vessel.
    pub fn best_available(&self) -> Option<(u64, f64)> {
        let mut available: Vec<_> = self.vessels.iter()
            .filter(|(_, v)| v.current_task.is_none() && v.health == VesselHealth::Nominal)
            .map(|(id, v)| (*id, v.trust))
            .collect();
        available.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        available.into_iter().next()
    }

    /// Fleet summary.
    pub fn status(&self) -> CaptainStatus {
        let active_missions = self.missions.values().filter(|m| m.status == MissionStatus::InProgress).count();
        let available = self.vessels.values().filter(|v| v.current_task.is_none()).count();
        CaptainStatus {
            name: self.name.clone(),
            confidence: self.confidence,
            total_vessels: self.vessels.len(),
            available,
            active_missions,
            total_missions: self.missions.len(),
            commands_issued: self.command_log.len(),
        }
    }

    pub fn mission(&self, id: u64) -> Option<&Mission> { self.missions.get(&id) }
    pub fn active_missions(&self) -> Vec<&Mission> {
        self.missions.values().filter(|m| m.status == MissionStatus::InProgress).collect()
    }
}

#[derive(Debug, Clone)]
pub enum CommandResult {
    Issued { target: u64, detail: String },
    Broadcast { recipients: usize, message: String },
    FleetWide { detail: String },
    Failed { reason: String },
    PolicySet,
    GroupFormed { name: String, members: usize, task: String },
    GroupDisbanded { name: String },
}

#[derive(Debug, Clone)]
pub struct CaptainStatus {
    pub name: String,
    pub confidence: f64,
    pub total_vessels: usize,
    pub available: usize,
    pub active_missions: usize,
    pub total_missions: usize,
    pub commands_issued: usize,
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_captain() -> Captain {
        let mut c = Captain::new("Kirk");
        c.enlist(1, "scout", 0.8);
        c.enlist(2, "worker", 0.6);
        c.enlist(3, "sensor", 0.7);
        c
    }

    #[test]
    fn test_enlist() {
        let c = make_captain();
        assert_eq!(c.vessels.len(), 3);
    }

    #[test]
    fn test_assign() {
        let mut c = make_captain();
        let result = c.command(FleetCommand::Assign { vessel_id: 1, task: "patrol".into(), priority: 1 });
        assert!(matches!(result, CommandResult::Issued { .. }));
    }

    #[test]
    fn test_assign_nonexistent() {
        let mut c = make_captain();
        let result = c.command(FleetCommand::Assign { vessel_id: 99, task: "x".into(), priority: 0 });
        assert!(matches!(result, CommandResult::Failed { .. }));
    }

    #[test]
    fn test_recall() {
        let mut c = make_captain();
        c.command(FleetCommand::Assign { vessel_id: 1, task: "patrol".into(), priority: 1 });
        let result = c.command(FleetCommand::Recall { vessel_id: 1 });
        assert!(matches!(result, CommandResult::Issued { .. }));
    }

    #[test]
    fn test_broadcast() {
        let mut c = make_captain();
        let result = c.command(FleetCommand::Broadcast { message: "attention".into() });
        assert!(matches!(result, CommandResult::Broadcast { recipients: 3, .. }));
    }

    #[test]
    fn test_allstop() {
        let mut c = make_captain();
        c.command(FleetCommand::Assign { vessel_id: 1, task: "a".into(), priority: 0 });
        c.command(FleetCommand::Assign { vessel_id: 2, task: "b".into(), priority: 0 });
        c.command(FleetCommand::AllStop);
        assert_eq!(c.vessels[&1].current_task, None);
        assert_eq!(c.vessels[&2].current_task, None);
    }

    #[test]
    fn test_mission_lifecycle() {
        let mut c = make_captain();
        let mid = c.plan_mission("explore", "Map sector 7").unwrap();
        c.launch_mission(mid, vec![1, 2]);
        let m = c.mission(mid).unwrap();
        assert_eq!(m.status, MissionStatus::InProgress);
    }

    #[test]
    fn test_subtask_completion() {
        let mut c = make_captain();
        let mid = c.plan_mission("build", "Construct outpost").unwrap();
        if let Some(m) = c.missions.get_mut(&mid) {
            m.add_subtask("survey", 1);
            m.add_subtask("build", 2);
            m.status = MissionStatus::InProgress;
        }
        c.complete_subtask(mid, "survey", "Area clear");
        let m = c.mission(mid).unwrap();
        assert_eq!(m.progress(), 0.5);
    }

    #[test]
    fn test_best_available() {
        let mut c = make_captain();
        c.command(FleetCommand::Assign { vessel_id: 1, task: "busy".into(), priority: 0 });
        let best = c.best_available().unwrap();
        assert_ne!(best.0, 1); // vessel 1 is busy
        assert_eq!(best.0, 3); // highest trust among available
    }

    #[test]
    fn test_max_concurrent_missions() {
        let mut c = make_captain();
        c.max_concurrent_missions = 1;
        c.plan_mission("a", "a").unwrap();
        let second = c.plan_mission("b", "b");
        assert!(second.is_none()); // at limit
    }

    #[test]
    fn test_captain_status() {
        let c = make_captain();
        let s = c.status();
        assert_eq!(s.total_vessels, 3);
        assert_eq!(s.available, 3);
        assert_eq!(s.active_missions, 0);
    }

    #[test]
    fn test_policy() {
        let mut c = make_captain();
        let result = c.command(FleetCommand::SetPolicy { key: "max_speed".into(), value: "10".into() });
        assert!(matches!(result, CommandResult::PolicySet));
    }
}
