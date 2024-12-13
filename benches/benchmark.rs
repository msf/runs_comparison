use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use runs_comparison::{BlockRangeSet, BlockRangesTrait, HashSetRanges, RangesVecOfStr, RangesStrOfArr, TreeSetRanges};
use std::fmt::Debug;
use serde::{Deserialize, Serialize};

// Helper function to generate test data
fn generate_sorted_unique_vec(range: std::ops::Range<u64>, gaps: &[u64]) -> Vec<u64> {
    (range.start..range.end)
        .filter(|x| !gaps.contains(x))
        .collect()
}

// Generic benchmark function for a single implementation
fn bench_implementation<T>(
    c: &mut Criterion,
    name: &str,
    group_name: &str,
    input_sizes: &[usize],
    test_fn: impl Fn(&T) -> (),
) where
    T: BlockRangesTrait + Clone + Debug + for<'a> Deserialize<'a> + Serialize,
{
    let mut group = c.benchmark_group(group_name);

    for &size in input_sizes {
        let gaps = if size >= 100_000 {
            vec![size as u64 / 2]
        } else {
            vec![]
        };
        let numbers = generate_sorted_unique_vec(1..(size as u64 + 1), &gaps);
        let implementation = T::from_sorted_vec(numbers.clone());

        group.bench_with_input(BenchmarkId::new(name, size), &size, |b, &_s| {
            b.iter(|| {
                test_fn(black_box(&implementation));
            })
        });
    }

    group.finish();
}

fn benchmark_conversion(c: &mut Criterion) {
    let sizes = [10_000, 1_000_000];
    
    for &size in &sizes {
        let gaps = if size >= 100_000 {
            vec![size as u64 / 2]
        } else {
            vec![]
        };
        let numbers = generate_sorted_unique_vec(1..(size as u64 + 1), &gaps);
        
        let mut group = c.benchmark_group("Conversion");
        
        // Benchmark each implementation using type-specific bench functions
        macro_rules! bench_conversion {
            ($name:expr, $type:ty) => {
                group.bench_with_input(
                    BenchmarkId::new($name, size), 
                    &size,
                    |b, &_s| {
                        b.iter(|| {
                            let result = <$type>::from_sorted_vec(black_box(numbers.clone()));
                            black_box(result)
                        })
                    }
                );
            };
        }

        bench_conversion!("RangesStrOfArr", RangesStrOfArr);
        bench_conversion!("RangesVecOfStr", RangesVecOfStr);
        bench_conversion!("BlockRangeSet", BlockRangeSet);
        bench_conversion!("HashSetRanges", HashSetRanges);
        bench_conversion!("TreeSetRanges", TreeSetRanges);
    }
}

fn benchmark_merge(c: &mut Criterion) {
    let sizes = [10_000, 1_000_000];
    
    macro_rules! bench_merge_impl {
        ($name:expr, $type:ty, $group:expr, $size:expr, $nums1:expr, $nums2:expr) => {
            let impl1 = <$type>::from_sorted_vec($nums1.clone());
            let impl2 = <$type>::from_sorted_vec($nums2.clone());
            
            $group.bench_with_input(
                BenchmarkId::new($name, $size),
                &$size,
                |b, &_s| {
                    b.iter(|| {
                        let mut temp = impl1.clone();
                        temp.merge(&impl2);
                        black_box(temp)
                    })
                }
            );
        };
    }

    let mut group = c.benchmark_group("Merge");

    for &size in &sizes {
        let gaps = (0..20).map(|i| (i + 1) * size as u64 / 20).collect::<Vec<u64>>();
        let range1 = 0..(2 * size as u64) / 3;
        let range2 = (size as u64) / 3..(2 * size as u64) / 3;
        let numbers1 = generate_sorted_unique_vec(range1, &gaps);
        let numbers2 = generate_sorted_unique_vec(range2, &gaps);

        bench_merge_impl!("RangesStrOfArr", RangesStrOfArr, group, size, numbers1, numbers2);
        bench_merge_impl!("RangesVecOfStr", RangesVecOfStr, group, size, numbers1, numbers2);
        bench_merge_impl!("BlockRangeSet", BlockRangeSet, group, size, numbers1, numbers2);
        bench_merge_impl!("HashSetRanges", HashSetRanges, group, size, numbers1, numbers2);
        bench_merge_impl!("TreeSetRanges", TreeSetRanges, group, size, numbers1, numbers2);
    }

    group.finish();
}

fn benchmark_find_missing(c: &mut Criterion) {
    let sizes = [10_000, 1_000_000];
    
    macro_rules! bench_find_missing_impl {
        ($name:expr, $type:ty, $group:expr, $size:expr, $nums:expr, $query:expr) => {
            let implementation = <$type>::from_sorted_vec($nums.clone());
            
            $group.bench_with_input(
                BenchmarkId::new($name, $size),
                &$size,
                |b, &_s| {
                    b.iter(|| {
                        let missing = implementation.find_missing(black_box(&$query));
                        black_box(missing)
                    })
                }
            );
        };
    }

    let mut group = c.benchmark_group("Find Missing");

    for &size in &sizes {
        let gaps = (0..10).map(|i| (i + 1) * size as u64 / 10).collect::<Vec<u64>>();
        let numbers = generate_sorted_unique_vec(1..(size as u64 + 1), &gaps);
        let mut rng = rand::thread_rng();
        let query: Vec<u64> = (0..(size / 100))
            .map(|_| rng.gen_range(1..=size as u64 * 2))
            .collect();

        bench_find_missing_impl!("RangesStrOfArr", RangesStrOfArr, group, size, numbers, query);
        bench_find_missing_impl!("RangesVecOfStr", RangesVecOfStr, group, size, numbers, query);
        bench_find_missing_impl!("BlockRangeSet", BlockRangeSet, group, size, numbers, query);
        bench_find_missing_impl!("HashSetRanges", HashSetRanges, group, size, numbers, query);
        bench_find_missing_impl!("TreeSetRanges", TreeSetRanges, group, size, numbers, query);
    }

    group.finish();
}

fn benchmark_serialization(c: &mut Criterion) {
    let sizes = [10_000, 1_000_000];
    
    macro_rules! bench_serialization_impl {
        ($name:expr, $type:ty, $group:expr, $size:expr, $nums:expr) => {
            let implementation = <$type>::from_sorted_vec($nums.clone());
            
            // Benchmark serialization
            $group.bench_with_input(
                BenchmarkId::new(format!("{}_to_bytes", $name), $size),
                &$size,
                |b, &_s| {
                    b.iter(|| {
                        let bytes = BlockRangesTrait::serialize(&implementation);
                        black_box(bytes)
                    })
                }
            );

            // Print size info
            let bytes = BlockRangesTrait::serialize(&implementation);
            let size_bytes = bytes.len();
            println!("{} with {} elements: {} bytes", $name, $size, size_bytes);
        };
    }

    let mut group = c.benchmark_group("Serialization");

    for &size in &sizes {
        let gaps = (0..50).map(|i| (i + 1) * size as u64 / 50).collect::<Vec<u64>>();
        let numbers = generate_sorted_unique_vec(1..(size as u64 + 1), &gaps);

        bench_serialization_impl!("RangesStrOfArr", RangesStrOfArr, group, size, numbers);
        bench_serialization_impl!("RangesVecOfStr", RangesVecOfStr, group, size, numbers);
        bench_serialization_impl!("BlockRangeSet", BlockRangeSet, group, size, numbers);
        bench_serialization_impl!("HashSetRanges", HashSetRanges, group, size, numbers);
        bench_serialization_impl!("TreeSetRanges", TreeSetRanges, group, size, numbers);
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_conversion,
    benchmark_merge,
    benchmark_find_missing,
    benchmark_serialization
);
criterion_main!(benches);
