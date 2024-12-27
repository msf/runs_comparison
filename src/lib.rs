use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};

/// Trait defining the Ranges API
pub trait BlockRangesTrait {
    /// Converts a sorted and deduplicated Vec<u64> into the Ranges structure
    fn from_sorted_vec(numbers: &[u64]) -> Self
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
pub struct BlockRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockRangeSet {
    pub(crate) ranges: Vec<BlockRange>,
}

impl BlockRangesTrait for BlockRangeSet {
    fn from_sorted_vec(input: &[u64]) -> Self {
        let mut set = Self { ranges: Vec::new() };
        if input.is_empty() {
            return set;
        }

        let mut numbers = input.to_vec();
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

impl BlockRangeSet {
    fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn ranges(&self) -> Vec<BlockRange> {
        self.ranges.clone()
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
    fn from_sorted_vec(numbers: &[u64]) -> Self {
        Self {
            numbers: numbers.to_vec().into_iter().collect(),
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
    fn from_sorted_vec(numbers: &[u64]) -> Self {
        Self {
            numbers: numbers.to_vec().into_iter().collect(),
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
            let implementation = T::from_sorted_vec(&case.numbers);
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
            let mut impl1 = T::from_sorted_vec(&nums1);
            let impl2 = T::from_sorted_vec(&nums2);
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
            let implementation = T::from_sorted_vec(&range_nums);
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
            let original = T::from_sorted_vec(&nums);
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
        let implementation = T::from_sorted_vec(&large_nums);
        assert_eq!(implementation.len(), 3);

        // Test with unordered input
        let unordered = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        let ordered = {
            let mut v = unordered.clone();
            v.sort_unstable();
            v
        };
        let impl_unordered = T::from_sorted_vec(&unordered);
        let impl_ordered = T::from_sorted_vec(&ordered);
        assert_eq!(impl_unordered.len(), impl_ordered.len());

        // Test with duplicates
        let with_duplicates = vec![1, 2, 2, 3, 3, 3, 4];
        let impl_duplicates = T::from_sorted_vec(&with_duplicates);
        assert_eq!(impl_duplicates.len(), 4);
    }

    // Run tests for all implementations
    #[test]
    fn test_block_range_set() {
        run_implementation_tests::<BlockRangeSet>();
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
            let implementation = T::from_sorted_vec(numbers);

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
        test_implementation::<HashSetRanges>("HashSetRange", &numbers);
        test_implementation::<TreeSetRanges>("TreeSetRange", &numbers);
    }
}
