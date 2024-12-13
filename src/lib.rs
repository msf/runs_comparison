use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashSet};

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

    /// Serializes the structure to bytes
    fn serialize(&self) -> Vec<u8>
    where
        Self: Serialize,
    {
        bincode::serialize(self).expect("Serialization failed")
    }

    /// Deserializes from bytes into the structure
    fn deserialize(data: &[u8]) -> Self
    where
        Self: Sized + for<'a> Deserialize<'a>,
    {
        bincode::deserialize(data).expect("Deserialization failed")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HashSetRanges {
    numbers: HashSet<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TreeSetRanges {
    numbers: BTreeSet<u64>,
}

impl BlockRangesTrait for HashSetRanges {
    fn from_sorted_vec(numbers: Vec<u64>) -> Self {
        Self {
            numbers: numbers.into_iter().collect(),
        }
    }

    fn merge(&mut self, other: &Self) {
        self.numbers.extend(other.numbers.iter().copied());
    }

    fn find_missing(&self, nums: &[u64]) -> Vec<u64> {
        nums.iter()
            .filter(|&&num| !self.numbers.contains(&num))
            .copied()
            .collect()
    }

    fn len(&self) -> usize {
        self.numbers.len()
    }
}

impl BlockRangesTrait for TreeSetRanges {
    fn from_sorted_vec(numbers: Vec<u64>) -> Self {
        Self {
            numbers: numbers.into_iter().collect(),
        }
    }

    fn merge(&mut self, other: &Self) {
        self.numbers.extend(other.numbers.iter().copied());
    }

    fn find_missing(&self, nums: &[u64]) -> Vec<u64> {
        nums.iter()
            .filter(|&&num| !self.numbers.contains(&num))
            .copied()
            .collect()
    }

    fn len(&self) -> usize {
        self.numbers.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockRangeSet {
    pub(crate) ranges: Vec<BlockRange>,
}

impl BlockRangesTrait for BlockRangeSet {
    fn from_sorted_vec(input: Vec<u64>) -> Self {
        let mut set = Self { ranges: Vec::new() };
        if input.is_empty() {
            return set;
        }

        let mut numbers = input.clone();
        numbers.sort_unstable();
        numbers.dedup();

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

/// Struct of Arrays (StrOfArr) Implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RangesStrOfArr {
    starts: Vec<u64>,
    counts: Vec<u32>,
}

impl BlockRangesTrait for RangesStrOfArr {
    fn from_sorted_vec(input: Vec<u64>) -> Self {
        let mut runs = RangesStrOfArr {
            starts: Vec::new(),
            counts: Vec::new(),
        };

        if input.is_empty() {
            return runs;
        }

        let mut numbers = input.clone();
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
                match num.cmp(run_start) {
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

    fn len(&self) -> usize {
        self.counts.iter().map(|&x| x as usize).sum()
    }
}

/// Array of Structs (VecOfStr) Implementation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Run {
    start: u64,
    count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RangesVecOfStr {
    runs: Vec<Run>,
}

impl BlockRangesTrait for RangesVecOfStr {
    fn from_sorted_vec(input: Vec<u64>) -> Self {
        let mut runs = RangesVecOfStr { runs: Vec::new() };

        if input.is_empty() {
            return runs;
        }

        let mut numbers = input.clone();
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
                let last_end = last.start + last.count as u64;
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

    fn len(&self) -> usize {
        self.runs.iter().map(|x| x.count as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    // Helper function to generate test data
    fn generate_sorted_unique_vec(range: std::ops::Range<u64>, gaps: &[u64]) -> Vec<u64> {
        (range.start..range.end)
            .filter(|x| !gaps.contains(x))
            .collect()
    }

    // Define test cases struct
    struct TestCase {
        name: &'static str,
        numbers: Vec<u64>,
        expected_ranges: Vec<(u64, u64)>, // (start, length) pairs
        expected_len: usize,
    }

    // Test cases that should work for all implementations
    fn get_test_cases() -> Vec<TestCase> {
        vec![
            TestCase {
                name: "basic sequence",
                numbers: vec![1, 2, 3, 5, 6, 8, 9, 10],
                expected_ranges: vec![(1, 3), (5, 2), (8, 3)],
                expected_len: 8,
            },
            TestCase {
                name: "single number",
                numbers: vec![42],
                expected_ranges: vec![(42, 1)],
                expected_len: 1,
            },
            TestCase {
                name: "empty input",
                numbers: vec![],
                expected_ranges: vec![],
                expected_len: 0,
            },
            TestCase {
                name: "non-sequential",
                numbers: vec![1, 3, 5, 7, 9],
                expected_ranges: vec![(1, 1), (3, 1), (5, 1), (7, 1), (9, 1)],
                expected_len: 5,
            },
        ]
    }

    // Generic test runner for any implementation of BlockRangesTrait
    fn run_implementation_tests<T>()
    where
        T: BlockRangesTrait
            + Clone
            + std::fmt::Debug
            + PartialEq
            + for<'a> Deserialize<'a>
            + Serialize,
    {
        // Run all test cases
        for case in get_test_cases() {
            let implementation = T::from_sorted_vec(case.numbers.clone());
            assert_eq!(
                implementation.len(),
                case.expected_len,
                "Length mismatch for test case: {}",
                case.name
            );
        }

        // Test merging
        test_merge::<T>();

        // Test finding missing numbers
        test_find_missing::<T>();

        // Test serialization
        test_serialization::<T>();

        // Test edge cases
        test_edge_cases::<T>();
    }

    fn test_merge<T>()
    where
        T: BlockRangesTrait + Clone + std::fmt::Debug + PartialEq,
    {
        let test_cases = vec![
            (vec![1, 2, 3], vec![4, 5, 6], 6, "Adjacent ranges"),
            (
                vec![1, 2, 3, 7, 8],
                vec![3, 4, 5, 8, 9],
                8,
                "Overlapping ranges",
            ),
            (vec![1, 2, 3], vec![5, 6, 7], 6, "Disjoint ranges"),
        ];

        for (nums1, nums2, expected_len, case_name) in test_cases {
            let mut impl1 = T::from_sorted_vec(nums1);
            let impl2 = T::from_sorted_vec(nums2);
            impl1.merge(&impl2);
            assert_eq!(
                impl1.len(),
                expected_len,
                "Merge length mismatch for case: {}",
                case_name
            );
        }
    }

    fn test_find_missing<T>()
    where
        T: BlockRangesTrait + Clone + std::fmt::Debug + PartialEq,
    {
        let test_cases = vec![
            (
                vec![1, 2, 3, 6, 7],
                vec![1, 2, 4, 5, 6, 8],
                vec![4, 5, 8],
                "Basic missing numbers",
            ),
            (vec![], vec![1, 2, 3], vec![1, 2, 3], "Empty range set"),
            (vec![1, 2, 3], vec![], vec![], "Empty query"),
        ];

        for (range_nums, query_nums, expected_missing, case_name) in test_cases {
            let implementation = T::from_sorted_vec(range_nums);
            let missing = implementation.find_missing(&query_nums);
            assert_eq!(
                missing, expected_missing,
                "Missing numbers mismatch for case: {}",
                case_name
            );
        }
    }

    fn test_serialization<T>()
    where
        T: BlockRangesTrait
            + Clone
            + std::fmt::Debug
            + PartialEq
            + for<'a> Deserialize<'a>
            + Serialize,
    {
        let test_cases = vec![vec![1, 2, 3, 5, 6], vec![], vec![42], vec![1, 3, 5, 7, 9]];

        for (i, nums) in test_cases.iter().enumerate() {
            let original = T::from_sorted_vec(nums.clone());
            let serialized = BlockRangesTrait::serialize(&original);
            let deserialized = BlockRangesTrait::deserialize(&serialized);
            assert_eq!(
                original, deserialized,
                "Serialization roundtrip failed for case {}",
                i
            );
        }
    }

    fn test_edge_cases<T>()
    where
        T: BlockRangesTrait + Clone + std::fmt::Debug + PartialEq,
    {
        // Test with large numbers
        let large_nums = vec![u64::MAX - 2, u64::MAX - 1, u64::MAX];
        let implementation = T::from_sorted_vec(large_nums.clone());
        assert_eq!(implementation.len(), 3);

        // Test with unordered input
        let unordered = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        let ordered = {
            let mut v = unordered.clone();
            v.sort_unstable();
            v
        };
        let impl_unordered = T::from_sorted_vec(unordered);
        let impl_ordered = T::from_sorted_vec(ordered);
        assert_eq!(impl_unordered.len(), impl_ordered.len());

        // Test with duplicates
        let with_duplicates = vec![1, 2, 2, 3, 3, 3, 4];
        let impl_duplicates = T::from_sorted_vec(with_duplicates);
        assert_eq!(impl_duplicates.len(), 4);
    }

    // Run tests for all implementations
    #[test]
    fn test_block_range_set() {
        run_implementation_tests::<BlockRangeSet>();
    }

    #[test]
    fn test_ranges_soa() {
        run_implementation_tests::<RangesStrOfArr>();
    }

    #[test]
    fn test_ranges_aos() {
        run_implementation_tests::<RangesVecOfStr>();
    }

    #[test]
    fn test_ranges_hash_set() {
        run_implementation_tests::<HashSetRanges>();
    }

    #[test]
    fn test_ranges_tree_set() {
        run_implementation_tests::<TreeSetRanges>();
    }

    // Performance test with large datasets
    #[test]
    fn test_large_dataset() {
        // Helper function to run test for a specific implementation
        fn test_implementation<T: BlockRangesTrait + std::fmt::Debug>(name: &str, numbers: &[u64])
        where
            T: BlockRangesTrait + for<'a> Deserialize<'a> + Serialize,
        {
            let implementation = T::from_sorted_vec(numbers.to_vec());

            // Test finding missing numbers
            let mut rng = rand::thread_rng();
            let query: Vec<u64> = (0..1000).map(|_| rng.gen_range(1..1_000_001)).collect();

            let missing = implementation.find_missing(&query);
            assert!(
                missing.len() <= query.len(),
                "{}: Found more missing numbers than queries",
                name
            );

            // Test serialization size
            let serialized = BlockRangesTrait::serialize(&implementation);
            println!("{} serialized size: {} bytes", name, serialized.len());
        }

        // Main test body
        let range = 1..1_000_001;
        let gaps = vec![50, 500_000, 600_000, 900_000];
        let numbers = generate_sorted_unique_vec(range, &gaps);

        test_implementation::<BlockRangeSet>("BlockRangeSet", &numbers);
        test_implementation::<RangesStrOfArr>("RangesStrOfArr", &numbers);
        test_implementation::<RangesVecOfStr>("RangesAoS", &numbers);
        test_implementation::<HashSetRanges>("HashSetRange", &numbers);
        test_implementation::<TreeSetRanges>("TreeSetRange", &numbers);
    }
}
