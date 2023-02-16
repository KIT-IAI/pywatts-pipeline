from pywatts_pipeline.core.steps.result_step import ResultStep
from pywatts_pipeline.core.steps.start_step import StartStep
from pywatts_pipeline.core.steps.step import Step
from pywatts_pipeline.core.steps.summary_step import SummaryStep


def ordered_step_list(steps, skip_steps):
    ordered_list = [list(filter(lambda x: x.last and x.name not in skip_steps, steps.values()))]
    finished = False
    while not finished:
        preds = []
        for s in ordered_list[-1]:
            preds.extend(get_preds(s, skip_steps))
        preds = set(preds)
        new_ordered_list = []
        for l in ordered_list:
            new_l = []
            for e in l:
                if not e in preds:
                    new_l.append(e)
            new_ordered_list.append(new_l)
        ordered_list = new_ordered_list
        if len(preds) == 0:
            finished = True
        else:
            ordered_list.append(preds)
    return ordered_list


def get_preds(step, skip_steps):
    preds = []
    for s in list(step.input_steps.values()) + (list(step.targets.values()) if hasattr(step, "targets") else []):
        if s.name in skip_steps:
            preds.extend(get_preds(s, skip_steps))
        else:
            preds.append(s)
    return preds


def visualise_pipeline(steps, skip_steps):
    from pywatts_pipeline.core.steps.pipeline_step import PipelineStep
    try:
        from schemdraw.util import Point
        from schemdraw import schemdraw, flow
    except ImportError as e:
        raise Exception("To visualise the pipeline, you need to install schemdraw") from e

    elements = {}
    w = 3
    h = 2
    with schemdraw.Drawing(show=False) as d:
        d.config(fontsize=10, unit=.25)
        for i, l in enumerate(ordered_step_list(steps, skip_steps)[::-1]):
            pos = Point(((w * 2) * i, -((len(steps) - 1) // 2) * (h * 2)))
            for j, s in enumerate(l):
                if isinstance(s, SummaryStep):
                    element = flow.RoundProcess(w=w, h=h)
                elif isinstance(s, PipelineStep):
                    element = flow.Subroutine(w=w, h=h)
                elif isinstance(s, Step):
                    element = flow.Process(w=w, h=h)
                elif isinstance(s, StartStep):
                    element = flow.Data(w=w, h=h)
                elif isinstance(s, ResultStep):
                    element = flow.Data(w=w, h=h)

                if pos is None:
                    element.label(s.name)
                else:
                    element.at(pos).label(
                        "\n".join(s.name[i:i+15] for i in range(0, len(s.name), 15)))
                d += element
                pos = pos + Point((0, h * 2))
                elements[s] = element, (j + 1) * (i + 1)

        # Find all successors:
        count = 0
        for l in ordered_step_list(steps, skip_steps)[::-1]:
            for s in l:
                preds = get_preds(s, skip_steps)
                for p in preds:
                    p_elem, p_j = elements[p]
                    s_elem, s_j = elements[s]
                    # Directly after each other
                    if p_elem.center.y == s_elem.center.y and (s_elem.center.x - p_elem.center.x) < w * 3:
                        d += flow.Arrow().at(p_elem.E).to(s_elem.W)
                    elif (s_elem.center.x - p_elem.center.x) < w * 3:
                        # In adjacent rows
                        d += flow.Wire("Z", k=w / 8, arrow="->").at(p_elem.E).to(s_elem.W)
                    else:
                        if p_elem.E.y < s_elem.E.y:
                            y = p_elem.E.y + .5 * h + .2 * h + (count - p_j) * .6 * h / count
                        else:
                            y = p_elem.E.y - .5 * h - .2 * h - (count - p_j) * .6 * h / count
                        d += flow.Line().at(p_elem.E). \
                            to(p_elem.E + Point((0.2 * w, 0)))
                        d += flow.Line().at(p_elem.E + Point((0.2 * w, 0))). \
                            to(Point((p_elem.E.x + w, y)))
                        d += flow.Line().at(Point((p_elem.E.x + w, y))). \
                            to(Point((s_elem.W.x - 0.8 * w, y)))
                        d += flow.Line().at(Point((s_elem.W.x - 0.8 * w, y))). \
                            to(Point((s_elem.W.x - 0.2 * w, s_elem.W.y)))
                        d += flow.Arrow().at(Point((s_elem.W.x - 0.2 * w, s_elem.W.y))). \
                            to(s_elem.W)
            count += len(l)
    return d
