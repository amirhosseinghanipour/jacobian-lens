use jacobian_lens::{Result, evaluation::*};
#[test]
fn released_eval_uses_item_macro_average_and_reports_token_coverage() -> Result<()> {
    let concept = |name: &str, rank: Option<usize>| ConceptRank {
        concept: name.into(),
        token_ids: if rank.is_some() { vec![1] } else { vec![] },
        minimum_rank_zero_based: rank,
    };
    let items = vec![
        ItemEvaluation {
            name: "one-concept".into(),
            readout_position: 3,
            concepts: vec![concept("Brazil", Some(0))],
        },
        ItemEvaluation {
            name: "three-concepts".into(),
            readout_position: 4,
            concepts: vec![
                concept("Spanish", Some(1)),
                concept("opposite", Some(40)),
                concept("multi-token", None),
            ],
        },
    ];
    let top1 = pass_at_k(&items, 1)?;
    assert_eq!(top1.pass_at_k, 0.5); // item macro average, not concept micro average
    assert_eq!(top1.concept_coverage, 0.75);
    let top2 = pass_at_k(&items, 2)?;
    assert!((top2.pass_at_k - 2. / 3.).abs() < 1e-12);
    let all = pass_at_k(&items, 100)?;
    assert!((all.pass_at_k - 5. / 6.).abs() < 1e-12); // uncovered concept cannot silently disappear
    Ok(())
}
