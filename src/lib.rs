use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Trait defining the Runs API
pub trait RunsTrait {
    /// Converts a sorted and deduplicated Vec<u64> into the Runs structure
    fn from_sorted_vec(numbers: Vec<u64>) -> Self
    where
        Self: Sized;

    /// Merges another Runs structure into self
    fn merge(&mut self, other: &Self);

    /// Finds missing numbers from the provided list that are not present in the Runs
    fn find_missing(&self, nums: &[u64]) -> Vec<u64>;

    /// Serializes the Runs structure into a binary format
    fn serialize(&self) -> Vec<u8>;

    /// Deserializes the Runs structure from a binary format
    fn deserialize(data: &[u8]) -> Self
    where
        Self: Sized;

    fn len(&self) -> usize;
}

/// Struct of Arrays (SoA) Implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunsSoA {
    starts: Vec<u64>,
    counts: Vec<u32>,
}

impl RunsTrait for RunsSoA {
    fn from_sorted_vec(mut numbers: Vec<u64>) -> Self {
        let mut runs = RunsSoA {
            starts: Vec::new(),
            counts: Vec::new(),
        };

        if numbers.is_empty() {
            return runs;
        }

        numbers.sort_unstable();
        numbers.dedup();

        let mut run_start = numbers[0];
        let mut run_length = 1;

        for &num in numbers.iter().skip(1) {
            if num == run_start + run_length as u64 {
                run_length += 1;
            } else {
                runs.starts.push(run_start);
                runs.counts.push(run_length);
                run_start = num;
                run_length = 1;
            }
        }

        // Push the last run
        runs.starts.push(run_start);
        runs.counts.push(run_length);

        runs
    }

    fn merge(&mut self, other: &Self) {
        let mut merged_starts = Vec::with_capacity(self.starts.len() + other.starts.len());
        let mut merged_counts = Vec::with_capacity(self.counts.len() + other.counts.len());

        let mut i = 0;
        let mut j = 0;

        // Merge the two sorted run lists
        while i < self.starts.len() && j < other.starts.len() {
            if self.starts[i] < other.starts[j] {
                merged_starts.push(self.starts[i]);
                merged_counts.push(self.counts[i]);
                i += 1;
            } else {
                merged_starts.push(other.starts[j]);
                merged_counts.push(other.counts[j]);
                j += 1;
            }
        }

        // Append any remaining runs
        while i < self.starts.len() {
            merged_starts.push(self.starts[i]);
            merged_counts.push(self.counts[i]);
            i += 1;
        }

        while j < other.starts.len() {
            merged_starts.push(other.starts[j]);
            merged_counts.push(other.counts[j]);
            j += 1;
        }

        // Now, merge overlapping or adjacent runs
        let mut reduced_starts = Vec::with_capacity(merged_starts.len());
        let mut reduced_counts = Vec::with_capacity(merged_counts.len());

        for idx in 0..merged_starts.len() {
            let current_start = merged_starts[idx];
            let current_count = merged_counts[idx];

            if reduced_starts.is_empty() {
                reduced_starts.push(current_start);
                reduced_counts.push(current_count);
            } else {
                let last_idx = reduced_starts.len() - 1;
                let last_start = reduced_starts[last_idx];
                let last_count = reduced_counts[last_idx];
                let last_end = last_start + last_count as u64;

                if current_start <= last_end {
                    // Overlapping or adjacent runs; merge them
                    let new_end = (current_start + current_count as u64).max(last_end);
                    reduced_counts[last_idx] = (new_end - last_start) as u32;
                } else {
                    // Non-overlapping run; add as new
                    reduced_starts.push(current_start);
                    reduced_counts.push(current_count);
                }
            }
        }

        // Update self with merged and reduced runs
        self.starts = reduced_starts;
        self.counts = reduced_counts;
    }

    fn find_missing(&self, nums: &[u64]) -> Vec<u64> {
        let mut missing = Vec::new();
        let mut run_iter = self.starts.iter().zip(self.counts.iter()).peekable();

        // Sort and deduplicate the input numbers for efficient traversal
        let mut sorted_nums = nums.to_vec();
        sorted_nums.sort_unstable();
        sorted_nums.dedup();

        for &num in &sorted_nums {
            // Advance the run iterator until the current run could contain the number
            while let Some(&(run_start, run_count)) = run_iter.peek() {
                let run_end = *run_start + (*run_count as u64);
                match num.cmp(&run_start) {
                    Ordering::Less => {
                        // Number is before the current run
                        missing.push(num);
                        break;
                    }
                    Ordering::Greater => {
                        if num >= run_end {
                            // Number is after the current run; advance
                            run_iter.next();
                        } else {
                            // Number is within the current run
                            break;
                        }
                    }
                    Ordering::Equal => {
                        // Number is exactly at the start of the run
                        break;
                    }
                }
            }

            // If no runs left, all remaining numbers are missing
            if run_iter.peek().is_none() && !missing.contains(&num) {
                missing.push(num);
            }
        }

        missing
    }

    fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Serialization failed")
    }

    fn deserialize(data: &[u8]) -> Self {
        bincode::deserialize(data).expect("Deserialization failed")
    }

    fn len(&self) -> usize {
        self.starts.len()
    }
}

/// Array of Structs (AoS) Implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Run {
    start: u64,
    count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunsAoS {
    runs: Vec<Run>,
}

impl RunsTrait for RunsAoS {
    fn from_sorted_vec(mut numbers: Vec<u64>) -> Self {
        let mut runs = RunsAoS { runs: Vec::new() };

        if numbers.is_empty() {
            return runs;
        }

        numbers.sort_unstable();
        numbers.dedup();

        let mut run_start = numbers[0];
        let mut run_length = 1;

        for &num in numbers.iter().skip(1) {
            if num == run_start + run_length as u64 {
                run_length += 1;
            } else {
                runs.runs.push(Run {
                    start: run_start,
                    count: run_length,
                });
                run_start = num;
                run_length = 1;
            }
        }

        // Push the last run
        runs.runs.push(Run {
            start: run_start,
            count: run_length,
        });

        runs
    }

    fn merge(&mut self, other: &Self) {
        let mut merged_runs = Vec::with_capacity(self.runs.len() + other.runs.len());

        let mut i = 0;
        let mut j = 0;

        // Merge the two sorted run lists
        while i < self.runs.len() && j < other.runs.len() {
            if self.runs[i].start < other.runs[j].start {
                merged_runs.push(self.runs[i].clone());
                i += 1;
            } else {
                merged_runs.push(other.runs[j].clone());
                j += 1;
            }
        }

        // Append any remaining runs
        while i < self.runs.len() {
            merged_runs.push(self.runs[i].clone());
            i += 1;
        }

        while j < other.runs.len() {
            merged_runs.push(other.runs[j].clone());
            j += 1;
        }

        // Now, merge overlapping or adjacent runs
        let mut reduced_runs: Vec<Run> = Vec::with_capacity(merged_runs.len());

        for run in merged_runs {
            if let Some(last) = reduced_runs.last_mut() {
                let last_end = (*last).start + (*last).count as u64;
                if run.start <= last_end {
                    // Overlapping or adjacent runs; merge them
                    let new_end = (run.start + run.count as u64).max(last_end);
                    last.count = (new_end - last.start) as u32;
                } else {
                    // Non-overlapping run; add as new
                    reduced_runs.push(run);
                }
            } else {
                reduced_runs.push(run);
            }
        }

        self.runs = reduced_runs;
    }

    fn find_missing(&self, nums: &[u64]) -> Vec<u64> {
        let mut missing = Vec::new();
        let mut run_iter = self.runs.iter().peekable();

        // Sort and deduplicate the input numbers for efficient traversal
        let mut sorted_nums = nums.to_vec();
        sorted_nums.sort_unstable();
        sorted_nums.dedup();

        for &num in &sorted_nums {
            // Advance the run iterator until the current run could contain the number
            while let Some(run) = run_iter.peek() {
                let run_end = run.start + run.count as u64;
                match num.cmp(&run.start) {
                    Ordering::Less => {
                        // Number is before the current run
                        missing.push(num);
                        break;
                    }
                    Ordering::Greater => {
                        if num >= run_end {
                            // Number is after the current run; advance
                            run_iter.next();
                        } else {
                            // Number is within the current run
                            break;
                        }
                    }
                    Ordering::Equal => {
                        // Number is exactly at the start of the run
                        break;
                    }
                }
            }

            // If no runs left, all remaining numbers are missing
            if run_iter.peek().is_none() && !missing.contains(&num) {
                missing.push(num);
            }
        }

        missing
    }

    fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Serialization failed")
    }

    fn deserialize(data: &[u8]) -> Self {
        bincode::deserialize(data).expect("Deserialization failed")
    }

    fn len(&self) -> usize {
        self.runs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Helper function to generate a sorted and deduplicated Vec<u64>
    fn generate_sorted_unique_vec(range: std::ops::Range<u64>, gaps: &[u64]) -> Vec<u64> {
        (range.start..range.end)
            .filter(|x| !gaps.contains(x))
            .collect()
    }

    /// Generic test for RunsTrait implementations
    fn run_tests<R: RunsTrait + PartialEq + std::fmt::Debug>() {
        // Test 1: Basic Conversion
        let numbers = generate_sorted_unique_vec(1..101, &[50]);
        let runs = R::from_sorted_vec(numbers.clone());

        // Expected Runs
        let expected_runs_soa = vec![
            (1, 49),  // 1-49
            (51, 50), // 51-100
        ];

        assert_eq!(runs.len(), expected_runs_soa.len());

        // Test 2: Merge Runs
        let numbers1 = generate_sorted_unique_vec(1..101, &[50]);
        let numbers2 = generate_sorted_unique_vec(101..201, &[150]);
        let mut runs1 = R::from_sorted_vec(numbers1.clone());
        let runs2 = R::from_sorted_vec(numbers2.clone());
        runs1.merge(&runs2);

        // Expected merged Runs
        let expected_merged_runs = vec![
            (1, 49),   // 1-49
            (51, 99),  // 51-149
            (151, 50), // 151-200
        ];

        assert_eq!(runs1.len(), expected_merged_runs.len());

        // Test 3: Find Missing Numbers
        let missing_numbers = vec![50, 100, 150, 200];
        let found_missing = runs1.find_missing(&missing_numbers);
        let expected_missing = vec![50, 150];

        assert_eq!(found_missing, expected_missing);

        // Test 4: Serialization and Deserialization
        let serialized = runs1.serialize();
        let deserialized = R::deserialize(&serialized);
        assert_eq!(runs1, deserialized);

        // Test 5: Edge Cases
        // Empty Runs
        let empty_runs = R::from_sorted_vec(Vec::new());
        let missing_empty = empty_runs.find_missing(&vec![1, 2, 3]);
        assert_eq!(missing_empty, vec![1, 2, 3]);

        // Single Element Runs
        let single_runs = R::from_sorted_vec(vec![10, 20, 30]);
        let missing_single = single_runs.find_missing(&vec![10, 15, 20, 25, 30, 35]);
        let expected_missing_single = vec![15, 25, 35];
        assert_eq!(missing_single, expected_missing_single);
    }

    #[test]
    fn test_runs_soa() {
        run_tests::<RunsSoA>();
    }

    #[test]
    fn test_runs_aos() {
        run_tests::<RunsAoS>();
    }

    /// Additional tests for merging edge cases
    #[test]
    fn test_merge_edge_cases() {
        // Implement only for RunsSoA and RunsAoS separately
        // Because they have different internal representations

        // RunsSoA Edge Cases
        {
            let numbers1 = vec![1, 2, 3, 4, 5];
            let numbers2 = vec![6, 7, 8, 9, 10];
            let mut runs1 = RunsSoA::from_sorted_vec(numbers1);
            let runs2 = RunsSoA::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.starts.len(), 1);
            assert_eq!(runs1.starts[0], 1);
            assert_eq!(runs1.counts[0], 10);
        }

        // RunsAoS Edge Cases
        {
            let numbers1 = vec![1, 2, 3, 4, 5];
            let numbers2 = vec![6, 7, 8, 9, 10];
            let mut runs1 = RunsAoS::from_sorted_vec(numbers1);
            let runs2 = RunsAoS::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.runs.len(), 1);
            assert_eq!(runs1.runs[0].start, 1);
            assert_eq!(runs1.runs[0].count, 10);
        }

        // Overlapping RunsSoA
        {
            let numbers1 = vec![1, 2, 3, 4, 5];
            let numbers2 = vec![4, 5, 6, 7, 8];
            let mut runs1 = RunsSoA::from_sorted_vec(numbers1);
            let runs2 = RunsSoA::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.starts.len(), 1);
            assert_eq!(runs1.starts[0], 1);
            assert_eq!(runs1.counts[0], 8);
        }

        // Overlapping RunsAoS
        {
            let numbers1 = vec![1, 2, 3, 4, 5];
            let numbers2 = vec![4, 5, 6, 7, 8];
            let mut runs1 = RunsAoS::from_sorted_vec(numbers1);
            let runs2 = RunsAoS::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.runs.len(), 1);
            assert_eq!(runs1.runs[0].start, 1);
            assert_eq!(runs1.runs[0].count, 8);
        }

        // Adjacent RunsSoA
        {
            let numbers1 = vec![1, 2, 3];
            let numbers2 = vec![4, 5, 6];
            let mut runs1 = RunsSoA::from_sorted_vec(numbers1);
            let runs2 = RunsSoA::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.starts.len(), 1);
            assert_eq!(runs1.starts[0], 1);
            assert_eq!(runs1.counts[0], 6);
        }

        // Adjacent RunsAoS
        {
            let numbers1 = vec![1, 2, 3];
            let numbers2 = vec![4, 5, 6];
            let mut runs1 = RunsAoS::from_sorted_vec(numbers1);
            let runs2 = RunsAoS::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.runs.len(), 1);
            assert_eq!(runs1.runs[0].start, 1);
            assert_eq!(runs1.runs[0].count, 6);
        }
    }

    /// Test find_missing with large input
    #[test]
    fn test_find_missing_large() {
        let range = 1..1_000_001;
        let gaps = vec![50, 500_000, 600_000, 900_000];
        let numbers = generate_sorted_unique_vec(range, &gaps);
        let runs_soa = RunsSoA::from_sorted_vec(numbers.clone());
        let runs_aos = RunsAoS::from_sorted_vec(numbers.clone());

        let mut missing_numbers = vec![50, 500_000, 600_000, 900_000];
        // add more random numbers to missing list to make the test harder
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let num = rng.gen_range(1..1_000_001);
            if gaps.contains(&num) {
                continue;
            }
            missing_numbers.push(num);
        }

        let missing_soa = runs_soa.find_missing(&missing_numbers);
        let missing_aos = runs_aos.find_missing(&missing_numbers);

        assert_eq!(missing_soa, gaps);
        assert_eq!(missing_aos, gaps);
    }
}
