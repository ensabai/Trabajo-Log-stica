from structure import solution

def improve(sol):
    improve = True
    while improve:
        improve = tryImprove(sol)


def tryImprove(sol):
    sel, ofVarSel, unsel, ofVarUnsel = selectInterchange(sol)
    if ofVarSel < ofVarUnsel:
        solution.removeFromSolution(sol, sel, ofVarSel)
        solution.addToSolution(sol, unsel, ofVarUnsel)
        return True
    return False


def selectInterchange(sol):
    n = sol['instance']['n']
    sel = -1
    bestSel = 0x3f3f3f
    unsel = -1
    bestUnsel = 0
    for v in sol['sol']:
        d = solution.distanceToSol(sol, v)
        if d < bestSel:
            bestSel = d
            sel = v
    for v in range(n):
        d = solution.distanceToSol(sol, v, without=sel)
        if not solution.contains(sol, v):
            if d > bestUnsel:
                bestUnsel = d
                unsel = v
    return sel, round(bestSel,2), unsel, round(bestUnsel,2)