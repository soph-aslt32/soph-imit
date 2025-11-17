use std::collections::HashMap;
use std::cmp;

/// 日付を表す構造体
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    pub year: u32,
    pub month: u32,
    pub day: u32,
}

impl Date {
    pub fn new(year: u32, month: u32, day: u32) -> Self {
        Self { year, month, day }
    }
}

/// タスクの状態を表す列挙型
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    NotStarted,
    InProgress,
    Completed,
    Cancelled,
}

/// ガントチャートのタスクを表す構造体
#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub name: String,
    pub start_date: Date,
    pub end_date: Date,
    pub status: TaskStatus,
    pub progress: f32, // 0.0 - 1.0
    pub dependencies: Vec<String>, // 依存するタスクのID
    pub assignee: Option<String>,
}

impl Task {
    pub fn new(id: String, name: String, start_date: Date, end_date: Date) -> Self {
        Self {
            id,
            name,
            start_date,
            end_date,
            status: TaskStatus::NotStarted,
            progress: 0.0,
            dependencies: Vec::new(),
            assignee: None,
        }
    }

    /// タスクの期間（日数）を計算
    pub fn duration_days(&self) -> u32 {
        // 簡単な日数計算（実際の実装では日付ライブラリを使用することを推奨）
        let start_days = self.start_date.year * 365 + self.start_date.month * 30 + self.start_date.day;
        let end_days = self.end_date.year * 365 + self.end_date.month * 30 + self.end_date.day;
        end_days.saturating_sub(start_days)
    }

    /// タスクの進捗を設定
    pub fn set_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
        
        // 進捗に応じてステータスを更新
        match self.progress {
            p if p == 0.0 => self.status = TaskStatus::NotStarted,
            p if p == 1.0 => self.status = TaskStatus::Completed,
            _ => self.status = TaskStatus::InProgress,
        }
    }

    /// 依存関係を追加
    pub fn add_dependency(&mut self, task_id: String) {
        if !self.dependencies.contains(&task_id) {
            self.dependencies.push(task_id);
        }
    }
}

/// ガントチャート全体を表す構造体
#[derive(Debug)]
pub struct GanttChart {
    pub title: String,
    pub tasks: HashMap<String, Task>,
    pub start_date: Option<Date>,
    pub end_date: Option<Date>,
}

impl GanttChart {
    pub fn new(title: String) -> Self {
        Self {
            title,
            tasks: HashMap::new(),
            start_date: None,
            end_date: None,
        }
    }

    /// タスクを追加
    pub fn add_task(&mut self, task: Task) {
        let task_id = task.id.clone();
        self.tasks.insert(task_id, task);
        self.update_date_range();
    }

    /// タスクを取得
    pub fn get_task(&self, id: &str) -> Option<&Task> {
        self.tasks.get(id)
    }

    /// タスクを可変参照で取得
    pub fn get_task_mut(&mut self, id: &str) -> Option<&mut Task> {
        self.tasks.get_mut(id)
    }

    /// タスクを削除
    pub fn remove_task(&mut self, id: &str) -> Option<Task> {
        let removed = self.tasks.remove(id);
        if removed.is_some() {
            self.update_date_range();
        }
        removed
    }

    /// 全タスクのリストを取得
    pub fn get_all_tasks(&self) -> Vec<&Task> {
        self.tasks.values().collect()
    }

    /// 日付範囲でタスクをフィルタリング
    pub fn get_tasks_in_range(&self, start: &Date, end: &Date) -> Vec<&Task> {
        self.tasks
            .values()
            .filter(|task| {
                task.start_date <= *end && task.end_date >= *start
            })
            .collect()
    }

    /// 担当者でタスクをフィルタリング
    pub fn get_tasks_by_assignee(&self, assignee: &str) -> Vec<&Task> {
        self.tasks
            .values()
            .filter(|task| {
                task.assignee.as_ref().map_or(false, |a| a == assignee)
            })
            .collect()
    }

    /// プロジェクト全体の進捗率を計算
    pub fn overall_progress(&self) -> f32 {
        if self.tasks.is_empty() {
            return 0.0;
        }

        let total_progress: f32 = self.tasks.values().map(|task| task.progress).sum();
        total_progress / self.tasks.len() as f32
    }

    /// 日付範囲を更新
    fn update_date_range(&mut self) {
        if self.tasks.is_empty() {
            self.start_date = None;
            self.end_date = None;
            return;
        }

        let mut min_date = None;
        let mut max_date = None;

        for task in self.tasks.values() {
            match min_date {
                None => min_date = Some(task.start_date.clone()),
                Some(ref date) if task.start_date < *date => {
                    min_date = Some(task.start_date.clone())
                }
                _ => {}
            }

            match max_date {
                None => max_date = Some(task.end_date.clone()),
                Some(ref date) if task.end_date > *date => {
                    max_date = Some(task.end_date.clone())
                }
                _ => {}
            }
        }

        self.start_date = min_date;
        self.end_date = max_date;
    }

    /// 依存関係の循環チェック
    pub fn has_circular_dependency(&self) -> bool {
        for task_id in self.tasks.keys() {
            if self.has_circular_dependency_recursive(task_id, &mut Vec::new()) {
                return true;
            }
        }
        false
    }

    fn has_circular_dependency_recursive(&self, task_id: &str, visited: &mut Vec<String>) -> bool {
        if visited.contains(&task_id.to_string()) {
            return true;
        }

        if let Some(task) = self.tasks.get(task_id) {
            visited.push(task_id.to_string());
            
            for dep_id in &task.dependencies {
                if self.has_circular_dependency_recursive(dep_id, visited) {
                    return true;
                }
            }
            
            visited.pop();
        }
        
        false
    }
}

/// ガントチャート間の距離を計算するためのメトリクス
#[derive(Debug, Clone)]
pub struct GanttDistance {
    pub temporal_distance: f32,
    pub structural_distance: f32,
    pub resource_distance: f32,
    pub overall_distance: f32,
}

impl GanttDistance {
    /// 複合距離を計算
    pub fn new(temporal: f32, structural: f32, resource: f32, weights: Option<(f32, f32, f32)>) -> Self {
        let (w1, w2, w3) = weights.unwrap_or((0.4, 0.4, 0.2));
        let overall = temporal * w1 + structural * w2 + resource * w3;
        
        Self {
            temporal_distance: temporal,
            structural_distance: structural,
            resource_distance: resource,
            overall_distance: overall,
        }
    }
}

/// ガントチャート距離計算器
pub struct GanttDistanceCalculator;

impl GanttDistanceCalculator {
    /// 2つのガントチャート間の距離を計算
    pub fn calculate_distance(chart1: &GanttChart, chart2: &GanttChart) -> GanttDistance {
        let temporal = Self::calculate_temporal_distance(chart1, chart2);
        let structural = Self::calculate_structural_distance(chart1, chart2);
        let resource = Self::calculate_resource_distance(chart1, chart2);
        
        GanttDistance::new(temporal, structural, resource, None)
    }
    
    /// 時間的距離を計算
    pub fn calculate_temporal_distance(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        if chart1.tasks.is_empty() || chart2.tasks.is_empty() {
            return 1.0; // 最大距離
        }
        
        let mut total_distance = 0.0;
        let mut task_count = 0;
        
        // プロジェクト全体の期間を取得して正規化の基準とする
        let max_project_span = Self::get_max_project_span(chart1, chart2);
        
        // 共通タスクの時間的差分を計算
        for (task_id, task1) in &chart1.tasks {
            if let Some(task2) = chart2.tasks.get(task_id) {
                let start_diff = Self::date_difference(&task1.start_date, &task2.start_date);
                let end_diff = Self::date_difference(&task1.end_date, &task2.end_date);
                let duration_diff = (task1.duration_days() as f32 - task2.duration_days() as f32).abs();
                
                // 最大プロジェクト期間で正規化
                let normalized_start_diff = if max_project_span > 0.0 { start_diff / max_project_span } else { 0.0 };
                let normalized_end_diff = if max_project_span > 0.0 { end_diff / max_project_span } else { 0.0 };
                let normalized_duration_diff = if max_project_span > 0.0 { duration_diff / max_project_span } else { 0.0 };
                
                // 0.0-1.0の範囲にクランプして平均
                let task_temporal_distance = (normalized_start_diff.min(1.0) + normalized_end_diff.min(1.0) + normalized_duration_diff.min(1.0)) / 3.0;
                total_distance += task_temporal_distance;
                task_count += 1;
            }
        }
        
        // プロジェクト期間の重複度を計算
        let overlap = Self::calculate_project_overlap(chart1, chart2);
        let temporal_alignment = 1.0 - overlap;
        
        // タスク存在差分のペナルティ
        let task_existence_penalty = Self::calculate_task_existence_penalty(chart1, chart2);
        
        if task_count > 0 {
            let avg_task_distance = total_distance / task_count as f32;
            // 3つの要素を平均して0.0-1.0の範囲に保つ
            (avg_task_distance + temporal_alignment + task_existence_penalty) / 3.0
        } else {
            1.0
        }
    }
    
    /// 構造的距離を計算（依存関係の類似性）
    pub fn calculate_structural_distance(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        let deps1 = Self::extract_dependencies(chart1);
        let deps2 = Self::extract_dependencies(chart2);
        
        // Jaccard距離を使用
        let intersection = deps1.intersection(&deps2).count();
        let union = deps1.union(&deps2).count();
        
        if union == 0 {
            0.0
        } else {
            1.0 - (intersection as f32 / union as f32)
        }
    }
    
    /// リソース配分距離を計算
    pub fn calculate_resource_distance(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        let assignees1: std::collections::HashSet<_> = chart1.tasks.values()
            .filter_map(|task| task.assignee.as_ref())
            .collect();
            
        let assignees2: std::collections::HashSet<_> = chart2.tasks.values()
            .filter_map(|task| task.assignee.as_ref())
            .collect();
        
        // 担当者の重複度
        let common_assignees = assignees1.intersection(&assignees2).count();
        let total_assignees = assignees1.union(&assignees2).count();
        
        // 進捗パターンの類似性
        let progress_similarity = Self::calculate_progress_similarity(chart1, chart2);
        
        let assignee_distance = if total_assignees > 0 {
            1.0 - (common_assignees as f32 / total_assignees as f32)
        } else {
            0.0
        };
        
        (assignee_distance + (1.0 - progress_similarity)) / 2.0
    }
    
    /// Edit Distance for Task Sequences（タスクシーケンスの編集距離）
    pub fn calculate_edit_distance(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        let seq1: Vec<_> = chart1.get_all_tasks()
            .iter()
            .map(|task| &task.id)
            .collect();
        let seq2: Vec<_> = chart2.get_all_tasks()
            .iter()
            .map(|task| &task.id)
            .collect();
        
        Self::levenshtein_distance(&seq1, &seq2) as f32 / 
            cmp::max(seq1.len(), seq2.len()).max(1) as f32
    }
    
    /// Dynamic Time Warping（動的時間伸縮法）による距離
    pub fn calculate_dtw_distance(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        let timeline1 = Self::create_timeline_vector(chart1);
        let timeline2 = Self::create_timeline_vector(chart2);
        
        if timeline1.is_empty() || timeline2.is_empty() {
            return 1.0;
        }
        
        let raw_dtw = Self::dtw(&timeline1, &timeline2);
        
        // タイムラインの最大値を取得して正規化
        let max_val1 = timeline1.iter().fold(0.0f32, |a, &b| a.max(b));
        let max_val2 = timeline2.iter().fold(0.0f32, |a, &b| a.max(b));
        let max_possible_diff = max_val1.max(max_val2);
        
        if max_possible_diff > 0.0 {
            (raw_dtw / max_possible_diff).min(1.0)
        } else {
            0.0
        }
    }
    
    // ヘルパーメソッド
    fn date_difference(date1: &Date, date2: &Date) -> f32 {
        let days1 = date1.year * 365 + date1.month * 30 + date1.day;
        let days2 = date2.year * 365 + date2.month * 30 + date2.day;
        (days1 as f32 - days2 as f32).abs()
    }
    
    fn calculate_project_overlap(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        let (start1, end1) = match (&chart1.start_date, &chart1.end_date) {
            (Some(s), Some(e)) => (s, e),
            _ => return 0.0,
        };
        let (start2, end2) = match (&chart2.start_date, &chart2.end_date) {
            (Some(s), Some(e)) => (s, e),
            _ => return 0.0,
        };
        
        let overlap_start = cmp::max(start1, start2);
        let overlap_end = cmp::min(end1, end2);
        
        if overlap_start <= overlap_end {
            let overlap_days = Self::date_difference(overlap_start, overlap_end);
            let total_span = Self::date_difference(
                cmp::min(start1, start2),
                cmp::max(end1, end2)
            );
            
            if total_span > 0.0 {
                overlap_days / total_span
            } else {
                1.0
            }
        } else {
            0.0
        }
    }
    
    fn calculate_task_existence_penalty(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        let tasks1: std::collections::HashSet<_> = chart1.tasks.keys().collect();
        let tasks2: std::collections::HashSet<_> = chart2.tasks.keys().collect();
        
        let common_tasks = tasks1.intersection(&tasks2).count();
        let total_unique_tasks = tasks1.union(&tasks2).count();
        
        if total_unique_tasks > 0 {
            1.0 - (common_tasks as f32 / total_unique_tasks as f32)
        } else {
            0.0
        }
    }
    
    fn extract_dependencies(chart: &GanttChart) -> std::collections::HashSet<(String, String)> {
        let mut deps = std::collections::HashSet::new();
        for task in chart.tasks.values() {
            for dep_id in &task.dependencies {
                deps.insert((task.id.clone(), dep_id.clone()));
            }
        }
        deps
    }
    
    fn calculate_progress_similarity(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        let mut total_diff = 0.0;
        let mut count = 0;
        
        for (task_id, task1) in &chart1.tasks {
            if let Some(task2) = chart2.tasks.get(task_id) {
                total_diff += (task1.progress - task2.progress).abs();
                count += 1;
            }
        }
        
        if count > 0 {
            1.0 - (total_diff / count as f32)
        } else {
            0.0
        }
    }
    
    fn levenshtein_distance<T: PartialEq>(seq1: &[T], seq2: &[T]) -> usize {
        let len1 = seq1.len();
        let len2 = seq2.len();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if seq1[i-1] == seq2[j-1] { 0 } else { 1 };
                matrix[i][j] = cmp::min(
                    cmp::min(matrix[i-1][j] + 1, matrix[i][j-1] + 1),
                    matrix[i-1][j-1] + cost
                );
            }
        }
        
        matrix[len1][len2]
    }
    
    fn create_timeline_vector(chart: &GanttChart) -> Vec<f32> {
        let mut timeline = vec![0.0; 365]; // 1年分のタイムライン
        
        for task in chart.tasks.values() {
            let start_day = (task.start_date.month - 1) * 30 + task.start_date.day - 1;
            let end_day = (task.end_date.month - 1) * 30 + task.end_date.day - 1;
            
            for day in start_day..=end_day.min(364) {
                timeline[day as usize] += task.progress;
            }
        }
        
        timeline
    }
    
    fn dtw(seq1: &[f32], seq2: &[f32]) -> f32 {
        let len1 = seq1.len();
        let len2 = seq2.len();
        
        if len1 == 0 || len2 == 0 {
            return 0.0;
        }
        
        let mut dtw_matrix = vec![vec![f32::INFINITY; len2]; len1];
        
        dtw_matrix[0][0] = (seq1[0] - seq2[0]).abs();
        
        for i in 1..len1 {
            dtw_matrix[i][0] = dtw_matrix[i-1][0] + (seq1[i] - seq2[0]).abs();
        }
        
        for j in 1..len2 {
            dtw_matrix[0][j] = dtw_matrix[0][j-1] + (seq1[0] - seq2[j]).abs();
        }
        
        for i in 1..len1 {
            for j in 1..len2 {
                let cost = (seq1[i] - seq2[j]).abs();
                dtw_matrix[i][j] = cost + dtw_matrix[i-1][j]
                    .min(dtw_matrix[i][j-1])
                    .min(dtw_matrix[i-1][j-1]);
            }
        }
        
        dtw_matrix[len1-1][len2-1]
    }
    
    fn get_max_project_span(chart1: &GanttChart, chart2: &GanttChart) -> f32 {
        let span1 = Self::get_project_span(chart1);
        let span2 = Self::get_project_span(chart2);
        span1.max(span2)
    }
    
    fn get_project_span(chart: &GanttChart) -> f32 {
        match (&chart.start_date, &chart.end_date) {
            (Some(start), Some(end)) => Self::date_difference(start, end),
            _ => {
                // start_date, end_dateがない場合は、タスクから計算
                if chart.tasks.is_empty() {
                    return 0.0;
                }
                let mut min_start = None;
                let mut max_end = None;
                
                for task in chart.tasks.values() {
                    match min_start {
                        None => min_start = Some(&task.start_date),
                        Some(current_min) if task.start_date < *current_min => {
                            min_start = Some(&task.start_date);
                        },
                        _ => {}
                    }
                    
                    match max_end {
                        None => max_end = Some(&task.end_date),
                        Some(current_max) if task.end_date > *current_max => {
                            max_end = Some(&task.end_date);
                        },
                        _ => {}
                    }
                }
                
                match (min_start, max_end) {
                    (Some(start), Some(end)) => Self::date_difference(start, end),
                    _ => 0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let start_date = Date::new(2024, 1, 1);
        let end_date = Date::new(2024, 1, 10);
        let task = Task::new("task1".to_string(), "テストタスク".to_string(), start_date, end_date);
        
        assert_eq!(task.id, "task1");
        assert_eq!(task.name, "テストタスク");
        assert_eq!(task.status, TaskStatus::NotStarted);
        assert_eq!(task.progress, 0.0);
    }

    #[test]
    fn test_gantt_chart() {
        let mut chart = GanttChart::new("テストプロジェクト".to_string());
        
        let task1 = Task::new(
            "task1".to_string(),
            "タスク1".to_string(),
            Date::new(2024, 1, 1),
            Date::new(2024, 1, 5)
        );
        
        let task2 = Task::new(
            "task2".to_string(),
            "タスク2".to_string(),
            Date::new(2024, 1, 6),
            Date::new(2024, 1, 10)
        );
        
        chart.add_task(task1);
        chart.add_task(task2);
        
        assert_eq!(chart.tasks.len(), 2);
        assert!(chart.get_task("task1").is_some());
        assert!(chart.get_task("task2").is_some());
    }

    #[test]
    fn test_progress_calculation() {
        let mut task = Task::new(
            "task1".to_string(),
            "テストタスク".to_string(),
            Date::new(2024, 1, 1),
            Date::new(2024, 1, 10)
        );
        
        task.set_progress(0.5);
        assert_eq!(task.progress, 0.5);
        assert_eq!(task.status, TaskStatus::InProgress);
        
        task.set_progress(1.0);
        assert_eq!(task.progress, 1.0);
        assert_eq!(task.status, TaskStatus::Completed);
    }
    
    #[test]
    fn test_gantt_distance_calculation() {
        let mut chart1 = GanttChart::new("プロジェクトA".to_string());
        let mut chart2 = GanttChart::new("プロジェクトB".to_string());
        
        // 同じタスクを追加
        let task1a = Task::new(
            "task1".to_string(),
            "設計".to_string(),
            Date::new(2024, 1, 1),
            Date::new(2024, 1, 10)
        );
        
        let mut task1b = Task::new(
            "task1".to_string(),
            "設計".to_string(),
            Date::new(2024, 1, 5),  // 異なる開始日
            Date::new(2024, 1, 15)  // 異なる終了日
        );
        task1b.set_progress(0.5);
        
        chart1.add_task(task1a);
        chart2.add_task(task1b);
        
        let distance = GanttDistanceCalculator::calculate_distance(&chart1, &chart2);
        
        // すべての距離は0.0-1.0の範囲内
        assert!(distance.temporal_distance >= 0.0 && distance.temporal_distance <= 1.0, 
                "Temporal distance {} is not in [0.0, 1.0]", distance.temporal_distance);
        assert!(distance.structural_distance >= 0.0 && distance.structural_distance <= 1.0,
                "Structural distance {} is not in [0.0, 1.0]", distance.structural_distance);
        assert!(distance.resource_distance >= 0.0 && distance.resource_distance <= 1.0,
                "Resource distance {} is not in [0.0, 1.0]", distance.resource_distance);
        assert!(distance.overall_distance >= 0.0 && distance.overall_distance <= 1.0,
                "Overall distance {} is not in [0.0, 1.0]", distance.overall_distance);
        
        // 時間的距離は0より大きい（日付が異なるため）
        assert!(distance.temporal_distance > 0.0);
        
        // 構造的距離は0（依存関係が同じため）
        assert_eq!(distance.structural_distance, 0.0);
    }
    
    #[test]
    fn test_structural_distance() {
        let mut chart1 = GanttChart::new("プロジェクトA".to_string());
        let mut chart2 = GanttChart::new("プロジェクトB".to_string());
        
        let task1 = Task::new("task1".to_string(), "タスク1".to_string(), 
                                   Date::new(2024, 1, 1), Date::new(2024, 1, 5));
        let mut task2 = Task::new("task2".to_string(), "タスク2".to_string(), 
                                   Date::new(2024, 1, 6), Date::new(2024, 1, 10));
        
        // chart1では task2 が task1 に依存
        task2.add_dependency("task1".to_string());
        
        chart1.add_task(task1.clone());
        chart1.add_task(task2.clone());
        
        // chart2では依存関係なし
        task2.dependencies.clear();
        chart2.add_task(task1);
        chart2.add_task(task2);
        
        let structural_distance = GanttDistanceCalculator::calculate_structural_distance(&chart1, &chart2);
        
        // 依存関係が異なるので距離は0より大きい
        assert!(structural_distance > 0.0);
    }
    
    #[test]
    fn test_edit_distance() {
        let mut chart1 = GanttChart::new("プロジェクトA".to_string());
        let mut chart2 = GanttChart::new("プロジェクトB".to_string());
        
        // 異なるタスクシーケンス
        chart1.add_task(Task::new("A".to_string(), "TaskA".to_string(), 
                                  Date::new(2024, 1, 1), Date::new(2024, 1, 5)));
        chart1.add_task(Task::new("B".to_string(), "TaskB".to_string(), 
                                  Date::new(2024, 1, 6), Date::new(2024, 1, 10)));
        
        chart2.add_task(Task::new("A".to_string(), "TaskA".to_string(), 
                                  Date::new(2024, 1, 1), Date::new(2024, 1, 5)));
        chart2.add_task(Task::new("C".to_string(), "TaskC".to_string(), 
                                  Date::new(2024, 1, 6), Date::new(2024, 1, 10)));
        
        let edit_distance = GanttDistanceCalculator::calculate_edit_distance(&chart1, &chart2);
        
        // 1つのタスクが異なるので編集距離は0より大きい
        assert!(edit_distance > 0.0);
        assert!(edit_distance <= 1.0);
    }
    
    #[test]
    fn test_distance_bounds_extreme_cases() {
        // 極端なケースでも距離が1.0を超えないことを確認
        
        // 1. 完全に異なるプロジェクト（何年も離れた日付）
        let mut chart1 = GanttChart::new("プロジェクトA".to_string());
        let mut chart2 = GanttChart::new("プロジェクトB".to_string());
        
        chart1.add_task(Task::new("task1".to_string(), "Task1".to_string(), 
                                  Date::new(2020, 1, 1), Date::new(2020, 12, 31)));
        chart2.add_task(Task::new("task1".to_string(), "Task1".to_string(), 
                                  Date::new(2030, 1, 1), Date::new(2030, 12, 31)));
        
        let distance = GanttDistanceCalculator::calculate_distance(&chart1, &chart2);
        assert!(distance.temporal_distance <= 1.0, "Extreme temporal distance exceeded 1.0: {}", distance.temporal_distance);
        assert!(distance.overall_distance <= 1.0, "Extreme overall distance exceeded 1.0: {}", distance.overall_distance);
        
        // 2. DTW距離のテスト
        let dtw_distance = GanttDistanceCalculator::calculate_dtw_distance(&chart1, &chart2);
        assert!(dtw_distance <= 1.0, "DTW distance exceeded 1.0: {}", dtw_distance);
        
        // 3. 空のチャート
        let empty_chart = GanttChart::new("Empty".to_string());
        let distance_with_empty = GanttDistanceCalculator::calculate_distance(&chart1, &empty_chart);
        assert!(distance_with_empty.temporal_distance <= 1.0);
        assert!(distance_with_empty.overall_distance <= 1.0);
        
        // 4. 完全一致のチャート
        let mut identical_chart1 = GanttChart::new("Test".to_string());
        let mut identical_chart2 = GanttChart::new("Test".to_string());
        
        let identical_task = Task::new("same".to_string(), "Same Task".to_string(),
                                      Date::new(2024, 1, 1), Date::new(2024, 1, 10));
        
        identical_chart1.add_task(identical_task.clone());
        identical_chart2.add_task(identical_task);
        
        let identical_distance = GanttDistanceCalculator::calculate_distance(&identical_chart1, &identical_chart2);
        assert_eq!(identical_distance.temporal_distance, 0.0, "Identical charts should have 0 temporal distance");
        assert_eq!(identical_distance.structural_distance, 0.0, "Identical charts should have 0 structural distance");
        assert_eq!(identical_distance.overall_distance, 0.0, "Identical charts should have 0 overall distance");
    }
}