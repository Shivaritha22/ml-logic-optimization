use egg::{rewrite as rw, *};
// define language

define_language! {
    enum BoolLang {
        "and" = And([Id; 2]),
        "or"  = Or([Id; 2]),
        "not" = Not([Id; 1]),
        Symbol(Symbol),
    }
}
fn main() {
    // declare rule list
    let rules: &[Rewrite<BoolLang, ()>] = &[
        // X AND TRUE = X
        rw!("and-true"; "(and ?x true)" => "?x"),
        // X AND FALSE = FALSE
        rw!("and-false"; "(and ?x false)" => "false"),
        // X AND X = X 
        rw!("and-same"; "(and ?x ?x)" => "?x"),
        // TRUE AND X = X
        rw!("true-and"; "(and true ?x)" => "?x"),
    ];
    // test expression:
    // A AND B AND TRUE AND D AND E AND TRUE AND G
    let expr = "(and A (and B (and true (and D (and E (and true G))))))";
    let runner = Runner::<BoolLang, (), ()>::default()
        .with_expr(&expr.parse().unwrap())
        .run(rules);
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (cost, simplified) = extractor.find_best(runner.roots[0]);
    println!("Original:   {}", expr);
    println!("Simplified: {}", simplified);
    println!("Cost (nodes): {}", cost);
}