    GraphType G;
    frontier old_frontier, new_frontier;
    api::edgemap( G,
	api::filter( api::src, api::weak, old_frontier ), // process edges (u,v) with u set in old_frontier ...
	api::filter( api::dst, api::weak, [&]( auto dst ) { } ), // ... and also lambda(v) must return true
	api::record( new_frontier, api::strong, api::reduction ), // set v in new_frontier if relax(u,v,e) returns true for any u,e
	api::relax( [&]( auto src, auto dst, auto edge ) { // execute this code for each processed (u,v)
			return src != dst; } ) ) // remember to return expression (syntax tree)
	.materialize(); // execute the map method
