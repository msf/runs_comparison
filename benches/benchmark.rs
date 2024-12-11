use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use runs_comparison::{RunsAoS, RunsSoA, RunsTrait};

// Helper function to generate a sorted and deduplicated Vec<u64> with specified gaps
fn generate_sorted_unique_vec(range: std::ops::Range<u64>, gaps: &[u64]) -> Vec<u64> {
    (range.start..range.end)
        .filter(|x| !gaps.contains(x))
        .collect()
}

fn benchmark_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Conversion");

    for size in &[10_000, 1_000_000] {
        let gaps = if *size >= 100_000 {
            vec![size / 2]
        } else {
            vec![]
        };
        let numbers = generate_sorted_unique_vec(1..(*size as u64)+1, &gaps);

        group.bench_with_input(BenchmarkId::new("RunsSoA", size), size, |b, &_s| {
            b.iter(|| {
                let runs = RunsSoA::from_sorted_vec(black_box(numbers.clone()));
                black_box(runs);
            })
        });

        group.bench_with_input(BenchmarkId::new("RunsAoS", size), size, |b, &_s| {
            b.iter(|| {
                let runs = RunsAoS::from_sorted_vec(black_box(numbers.clone()));
                black_box(runs);
            })
        });
    }

    group.finish();
}

fn benchmark_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("Merge");

    for size in &[10_000, 1_000_000] {
        // generate 20 gaps
        let gaps = (0..20).map(|i| (i + 1) * size / 20).collect::<Vec<u64>>();

        // generate two ranges that overlap by 1/3rd
        let range1 = 0..(2 * *size as u64) / 3;
        let range2 = (*size as u64) / 3..(2 * *size as u64) / 3;

        let numbers1 = generate_sorted_unique_vec(range1, &gaps);
        let numbers2 = generate_sorted_unique_vec(range2, &gaps);

        let runs1_soa = RunsSoA::from_sorted_vec(numbers1.clone());
        let runs2_soa = RunsSoA::from_sorted_vec(numbers2.clone());

        let runs1_aos = RunsAoS::from_sorted_vec(numbers1.clone());
        let runs2_aos = RunsAoS::from_sorted_vec(numbers2.clone());

        group.bench_with_input(BenchmarkId::new("RunsSoA", size), size, |b, &_s| {
            b.iter(|| {
                let mut runs1 = runs1_soa.clone();
                runs1.merge(&runs2_soa);
                black_box(runs1);
            })
        });

        group.bench_with_input(BenchmarkId::new("RunsAoS", size), size, |b, &_s| {
            b.iter(|| {
                let mut runs1 = runs1_aos.clone();
                runs1.merge(&runs2_aos);
                black_box(runs1);
            })
        });
    }

    group.finish();
}

fn benchmark_find_missing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Find Missing");

    for size in &[10_000, 1_000_000] {
        // generate 10 gaps with the middle gap being the size / 2
        let gaps = (0..10).map(|i| (i + 1) * size / 10).collect::<Vec<u64>>();
        let numbers = generate_sorted_unique_vec(1..(*size as u64)+1, &gaps);
        let runs_soa = RunsSoA::from_sorted_vec(numbers.clone());
        let runs_aos = RunsAoS::from_sorted_vec(numbers.clone());

        // Generate missing numbers
        let mut rng = rand::thread_rng();
        let missing: Vec<u64> = (0..(size / 100))
            .map(|_| rng.gen_range(1..=(*size as u64) * 2))
            .collect();

        group.bench_with_input(BenchmarkId::new("RunsSoA", size), size, |b, &_s| {
            b.iter(|| {
                let missing_found = runs_soa.find_missing(black_box(&missing));
                black_box(missing_found);
            })
        });

        group.bench_with_input(BenchmarkId::new("RunsAoS", size), size, |b, &_s| {
            b.iter(|| {
                let missing_found = runs_aos.find_missing(black_box(&missing));
                black_box(missing_found);
            })
        });
    }

    group.finish();
}

fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Serialization");

    for size in &[10_000, 1_000_000] {
        let gaps = (0..50).map(|i| (i + 1) * size / 50).collect::<Vec<u64>>();
        let numbers = generate_sorted_unique_vec(1..(*size as u64)+1, &gaps);
        let runs_soa = RunsSoA::from_sorted_vec(numbers.clone());
        let runs_aos = RunsAoS::from_sorted_vec(numbers.clone());

        group.bench_with_input(BenchmarkId::new("RunsSoA", size), size, |b, &_s| {
            b.iter(|| {
                let serialized = runs_soa.serialize();
                black_box(serialized);
            })
        });

        group.bench_with_input(BenchmarkId::new("RunsAoS", size), size, |b, &_s| {
            b.iter(|| {
                let serialized = runs_aos.serialize();
                black_box(serialized);
            })
        });
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
