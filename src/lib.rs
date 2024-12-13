use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BlockRangeSet {
    pub(crate) ranges: Vec<BlockRange>,
}

/// Trait defining the Ranges API
pub trait BlockRangesTrait {
    /// Converts a sorted and deduplicated Vec<u64> into the Ranges structure
    fn from_sorted_vec(numbers: Vec<u64>) -> Self
    where
        Self: Sized;

    /// Merges another Ranges structure into self
    fn merge(&mut self, other: &Self);

    /// Finds missing numbers from the provided list that are not present in the Ranges
    fn find_missing(&self, nums: &[u64]) -> Vec<u64>;

    fn len(&self) -> usize;
}

impl BlockRangesTrait for BlockRangeSet {
    fn from_sorted_vec(numbers: Vec<u64>) -> Self {
        let mut set = Self { ranges: Vec::new() };
        if numbers.is_empty() {
            return set;
        }

        let mut current = BlockRange {
            start: numbers[0],
            end: numbers[0],
        };

        for &num in numbers.iter().skip(1) {
            if num == current.end + 1 {
                current.end = num;
            } else {
                set.ranges.push(current);
                current = BlockRange {
                    start: num,
                    end: num,
                };
            }
        }
        set.ranges.push(current);
        set
    }

    fn merge(&mut self, other: &Self) {
        if other.ranges.is_empty() {
            return;
        }
        if self.ranges.is_empty() {
            self.ranges = other.ranges.clone();
            return;
        }

        // First pass: merge both ordered lists into a new Vec
        let mut merged = Vec::with_capacity(self.ranges.len() + other.ranges.len());
        let mut i = 0;
        let mut j = 0;

        while i < self.ranges.len() && j < other.ranges.len() {
            if self.ranges[i].start < other.ranges[j].start {
                merged.push(self.ranges[i].clone());
                i += 1;
            } else {
                merged.push(other.ranges[j].clone());
                j += 1;
            }
        }

        // Add remaining ranges
        merged.extend_from_slice(&self.ranges[i..]);
        merged.extend_from_slice(&other.ranges[j..]);

        // Second pass: reduce overlapping ranges
        let mut reduced = Vec::with_capacity(merged.len());
        let mut current = merged[0].clone();

        for next in merged.iter().skip(1) {
            if next.start <= current.end + 1 {
                // Ranges overlap or are adjacent
                current.end = current.end.max(next.end);
            } else {
                reduced.push(current);
                current = next.clone();
            }
        }
        reduced.push(current);

        self.ranges = reduced;
    }

    fn find_missing(&self, nums: &[u64]) -> Vec<u64> {
        if self.ranges.is_empty() {
            return nums.to_vec();
        }

        nums.iter()
            .filter(|&&num| {
                let pos = self.ranges.partition_point(|r| r.end < num);
                pos >= self.ranges.len() || self.ranges[pos].start > num
            })
            .copied()
            .collect()
    }

    fn len(&self) -> usize {
        self.ranges
            .iter()
            .map(|range| (range.end - range.start + 1) as usize)
            .sum()
    }
}

/// Struct of Arrays (SoA) Implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RangesSoA {
    starts: Vec<u64>,
    counts: Vec<u32>,
}

impl BlockRangesTrait for RangesSoA {
    fn from_sorted_vec(mut numbers: Vec<u64>) -> Self {
        let mut runs = RangesSoA {
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
pub struct RangesAoS {
    runs: Vec<Run>,
}

impl BlockRangesTrait for RangesAoS {
    fn from_sorted_vec(mut numbers: Vec<u64>) -> Self {
        let mut runs = RangesAoS { runs: Vec::new() };

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
    use bincode;
    use rand::Rng;

    /// Helper function to generate a sorted and deduplicated Vec<u64>
    fn generate_sorted_unique_vec(range: std::ops::Range<u64>, gaps: &[u64]) -> Vec<u64> {
        (range.start..range.end)
            .filter(|x| !gaps.contains(x))
            .collect()
    }

    /// Generic test for BlockRangesTrait implementations
    fn run_tests<R: BlockRangesTrait + PartialEq + std::fmt::Debug>() {
        // Test 1: Basic Conversion
        let numbers = generate_sorted_unique_vec(1..101, &[50]);
        let runs = R::from_sorted_vec(numbers.clone());

        // Expected Ranges
        let expected_runs_soa = vec![
            (1, 49),  // 1-49
            (51, 50), // 51-100
        ];

        assert_eq!(runs.len(), expected_runs_soa.len());

        // Test 2: Merge Ranges
        let numbers1 = generate_sorted_unique_vec(1..101, &[50]);
        let numbers2 = generate_sorted_unique_vec(101..201, &[150]);
        let mut runs1 = R::from_sorted_vec(numbers1.clone());
        let runs2 = R::from_sorted_vec(numbers2.clone());
        runs1.merge(&runs2);

        // Expected merged Ranges
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
        // Empty Ranges
        let empty_runs = R::from_sorted_vec(Vec::new());
        let missing_empty = empty_runs.find_missing(&vec![1, 2, 3]);
        assert_eq!(missing_empty, vec![1, 2, 3]);

        // Single Element Ranges
        let single_runs = R::from_sorted_vec(vec![10, 20, 30]);
        let missing_single = single_runs.find_missing(&vec![10, 15, 20, 25, 30, 35]);
        let expected_missing_single = vec![15, 25, 35];
        assert_eq!(missing_single, expected_missing_single);
    }

    #[test]
    fn test_runs_soa() {
        run_tests::<RangesSoA>();
    }

    #[test]
    fn test_runs_aos() {
        run_tests::<RangesAoS>();
    }

    /// Additional tests for merging edge cases
    #[test]
    fn test_merge_edge_cases() {
        // Implement only for RangesSoA and RangesAoS separately
        // Because they have different internal representations

        // RangesSoA Edge Cases
        {
            let numbers1 = vec![1, 2, 3, 4, 5];
            let numbers2 = vec![6, 7, 8, 9, 10];
            let mut runs1 = RangesSoA::from_sorted_vec(numbers1);
            let runs2 = RangesSoA::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.starts.len(), 1);
            assert_eq!(runs1.starts[0], 1);
            assert_eq!(runs1.counts[0], 10);
        }

        // RangesAoS Edge Cases
        {
            let numbers1 = vec![1, 2, 3, 4, 5];
            let numbers2 = vec![6, 7, 8, 9, 10];
            let mut runs1 = RangesAoS::from_sorted_vec(numbers1);
            let runs2 = RangesAoS::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.runs.len(), 1);
            assert_eq!(runs1.runs[0].start, 1);
            assert_eq!(runs1.runs[0].count, 10);
        }

        // Overlapping RangesSoA
        {
            let numbers1 = vec![1, 2, 3, 4, 5];
            let numbers2 = vec![4, 5, 6, 7, 8];
            let mut runs1 = RangesSoA::from_sorted_vec(numbers1);
            let runs2 = RangesSoA::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.starts.len(), 1);
            assert_eq!(runs1.starts[0], 1);
            assert_eq!(runs1.counts[0], 8);
        }

        // Overlapping RangesAoS
        {
            let numbers1 = vec![1, 2, 3, 4, 5];
            let numbers2 = vec![4, 5, 6, 7, 8];
            let mut runs1 = RangesAoS::from_sorted_vec(numbers1);
            let runs2 = RangesAoS::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.runs.len(), 1);
            assert_eq!(runs1.runs[0].start, 1);
            assert_eq!(runs1.runs[0].count, 8);
        }

        // Adjacent RangesSoA
        {
            let numbers1 = vec![1, 2, 3];
            let numbers2 = vec![4, 5, 6];
            let mut runs1 = RangesSoA::from_sorted_vec(numbers1);
            let runs2 = RangesSoA::from_sorted_vec(numbers2);
            runs1.merge(&runs2);

            assert_eq!(runs1.starts.len(), 1);
            assert_eq!(runs1.starts[0], 1);
            assert_eq!(runs1.counts[0], 6);
        }

        // Adjacent RangesAoS
        {
            let numbers1 = vec![1, 2, 3];
            let numbers2 = vec![4, 5, 6];
            let mut runs1 = RangesAoS::from_sorted_vec(numbers1);
            let runs2 = RangesAoS::from_sorted_vec(numbers2);
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
        let runs_soa = RangesSoA::from_sorted_vec(numbers.clone());
        let runs_aos = RangesAoS::from_sorted_vec(numbers.clone());

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
    #[test]
    fn test_from_sorted_vec() {
        let nums = vec![1, 2, 3, 5, 6, 8, 9, 10];
        let set = BlockRangeSet::from_sorted_vec(nums);
        assert_eq!(set.ranges.len(), 3);
        assert_eq!(set.ranges[0].start, 1);
        assert_eq!(set.ranges[0].end, 3);
        assert_eq!(set.ranges[1].start, 5);
        assert_eq!(set.ranges[1].end, 6);
        assert_eq!(set.ranges[2].start, 8);
        assert_eq!(set.ranges[2].end, 10);
    }

    #[test]
    fn test_merge() {
        let mut set1 = BlockRangeSet::from_sorted_vec(vec![1, 2, 3, 7, 8]);
        let set2 = BlockRangeSet::from_sorted_vec(vec![3, 4, 5, 8, 9]);
        set1.merge(&set2);
        assert_eq!(set1.ranges.len(), 2);
        assert_eq!(set1.ranges[0].start, 1);
        assert_eq!(set1.ranges[0].end, 5);
        assert_eq!(set1.ranges[1].start, 7);
        assert_eq!(set1.ranges[1].end, 9);
    }

    #[test]
    fn test_find_missing() {
        let set = BlockRangeSet::from_sorted_vec(vec![1, 2, 3, 6, 7]);
        let missing = set.find_missing(&[1, 2, 4, 5, 6, 8]);
        assert_eq!(missing, vec![4, 5, 8]);
    }

    #[test]
    fn test_len() {
        let set = BlockRangeSet::from_sorted_vec(vec![1, 2, 3, 5, 6]);
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_serialization() {
        let set = BlockRangeSet::from_sorted_vec(vec![1, 2, 3, 5, 6]);
        let serialized = bincode::serialize(&set).expect("Failed to serialize");
        let deserialized: BlockRangeSet =
            bincode::deserialize(&serialized).expect("Failed to deserialize");
        assert_eq!(deserialized.len(), set.len());
    }
}
