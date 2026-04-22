use egg::{rewrite as rw, *};

define_language! {
    enum BoolLang {
        "and" = And([Id; 2]),
        "or"  = Or([Id; 2]),
        "not" = Not([Id; 1]),
        Symbol(Symbol),
    }
}

fn main() {
    // Boolean rewriting rules
    let rules: &[Rewrite<BoolLang, ()>] = &[
        rw!("and-comm-lr"; "(and ?x ?y)" => "(and ?y ?x)"),
        rw!("and-comm-rl"; "(and ?y ?x)" => "(and ?x ?y)"),
        rw!("and-true";  "(and ?x true)"  => "?x"),
        rw!("and-false"; "(and ?x false)" => "false"),
        rw!("and-same";  "(and ?x ?x)"    => "?x"),
        rw!("true-and";  "(and true ?x)"  => "?x"),
    ];

    // Decisions classified as TRUE from profiling data:
//   C = TRUE (Pnet Nms Cross - 0.0% avg rejection, 100.0% zero across 1000 images)

    // Decisions classified as VARIABLE (kept):
//   A = VARIABLE (Pnet Threshold - 98.0% avg rejection)
//   B = VARIABLE (Pnet Nms Scale - 59.4% avg rejection)
//   D = VARIABLE (Rnet Threshold - 70.8% avg rejection)
//   E = VARIABLE (Rnet Nms - 33.6% avg rejection)
//   F = MOSTLY TRUE (Onet Threshold - 7.4% avg rejection)
//   G = VARIABLE (Onet Nms - 83.2% avg rejection)

    // Original formula with TRUE substituted for redundant decisions:
    // A AND B AND TRUE AND D AND E AND F AND G
    let expr = "(and A (and B (and true (and D (and E (and F G))))))";

    println!("Running egg equality saturation...");
    println!("Input:  {}", expr);

    let runner = Runner::<BoolLang, (), ()>::default()
        .with_expr(&expr.parse().unwrap())
        .run(rules);

    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (cost, simplified) = extractor.find_best(runner.roots[0]);

    println!("Output: {}", simplified);
    println!("Nodes:  {}", cost);
    println!();
    println!("This proves:");
    println!("  A AND B AND TRUE AND D AND E AND F AND G");
    println!("  = A AND B AND D AND E AND F AND G");
    println!("  via boolean identity laws");
}
