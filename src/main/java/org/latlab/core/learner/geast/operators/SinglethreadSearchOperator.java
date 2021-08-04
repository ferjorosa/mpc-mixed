package org.latlab.core.learner.geast.operators;

import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.util.Evaluator;

import java.util.LinkedList;

/**
 * Search operator that perform candidate search sequentially using only one
 * thread. However, the estimation on the candidates may use multiple threads.
 * 
 * @author leonard
 * 
 */
public abstract class SinglethreadSearchOperator extends SearchOperator {

	public SinglethreadSearchOperator(ISearchOperatorContext context) {
		super(context);
	}

	/**
	 * Returns the {@code base} candidate if it can't find a better one. So it
	 * guarantees to return the best model (either the base model or a found
	 * better model).
	 * 
	 * @param base
	 *            base model to search from
	 * @return the best candidate it can find
	 */
	@Override
	public SearchCandidate search(IModelWithScore base,
			Evaluator<SearchCandidate> evaluator) {
		context.log().writeStartElementWithTime(name(), null);

		LinkedList<SearchCandidate> candidates = generateCandidates(base);

		// here screens the generated candidates by adding them to a queue with
		// bounded size
		ScreenQueue screenQueue = new ScreenQueue(context.screeningSize());

		// estimates the candidates one by one
		while (!candidates.isEmpty()) {
			SearchCandidate candidate = null;
			try {
				candidate = candidates.removeFirst();
				candidate.evaluate(context.screeningEm(), evaluator);
				screenQueue.add(candidate);
				log(candidate);
			} catch (Exception e) {
				context.log().write(e, candidate);
			}
		}

		candidates.clear();

		// evaluate more carefully on the remaining candidates to select the
		// best one
		SearchCandidate best = new GivenCandidate(base);

		for (SearchCandidate candidate : screenQueue) {
			try {
				candidate.evaluate(context.selectionEm(), evaluator);

				// log the original model's name, so that it can be compared
				// with the candidates generated by the search operator.
				context.log().writeElement("selecting", candidate, true);

				// the BIC comparator is in descending order
				if (SearchCandidate.SCORE_COMPARATOR.compare(candidate, best) < 0)
					best = candidate;
			} catch (Exception e) {
				context.log().write(e, candidate);
			}
		}

		context.log().writeElement("completed", best, true);
		context.log().writeEndElement(name());

		return best;
	}

}