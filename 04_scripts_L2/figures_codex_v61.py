"""Export Codex v6.1 figure variants from the four mechanism axes and v6 vector artwork."""
from shutil import copyfile
import pymupdf as pdf
from matplotlib import pyplot as plt
from fig2_decay_mechanisms import build_figure
from paths_common import FIGURES

def diffraction():
    copyfile(FIGURES/'fig4_fixed_entropy_Codex_v6.pdf', FIGURES/'fig4_fixed_entropy_Codex_v61.pdf')
    with pdf.open(FIGURES/'S11_metric_tension_Codex_v6.pdf') as doc:
        page=doc[0]
        tags={w[4]:pdf.Rect(w[:4]) for w in page.get_text('words') if w[4] in ('e','f','g') and w[1]<5}
        assert set(tags)==set('efg')
        for box in tags.values():page.add_redact_annot(box, fill=False)
        page.apply_redactions(images=0,graphics=0)
        for old,new in zip('efg','abc'):
            box=tags[old]
            page.insert_text((box.x0,box.y1-2.5),new,fontname='hebo',fontsize=12)
        doc.save(FIGURES/'S11_metric_tension_Codex_v61.pdf',deflate=True,garbage=4)

if __name__=='__main__':
    figure=build_figure()
    figure.savefig(FIGURES/'fig2_constraints_Codex_v61.pdf', bbox_inches='tight')
    plt.close(figure)
    diffraction()
