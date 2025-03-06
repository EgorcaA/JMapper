import numpy as np
import pickle 
from  tqdm import tqdm
import plotly.graph_objects as go


import numpy as np
from scipy.spatial import Voronoi

def plot_Jq_1BZ(calc, kpoints_2BZ, z):
    '''
    kpoints_2BZ in fractional coordinates
    z - J(q) values corresponding to kpoints_2BZ
    calc - instance of qe class
    '''
    
    acell = np.linalg.norm(calc.acell[0])
    b1 = calc.bcell[0][:2] / (2. * np.pi / acell)  # First reciprocal lattice vector
    b2 = calc.bcell[1][:2] / (2. * np.pi / acell) # Second reciprocal lattice vector

    # recip cart coords of vectors
    coords = [ kpt[0]  * b1 + kpt[1]* b2 for kpt in kpoints_2BZ]
    coords = np.array(coords)
    kx = coords[:, 0] 
    ky = coords[:, 1] 


    fig = go.Figure()
    fig.add_trace(go.Contour(x=kx,y=ky,z=z,
                            line_smoothing=1.3,
                            colorbar=dict(
                            title="Z Value", 
                            x=1.05,  # Move colorbar to the right
                            len=0.75  # Shorten colorbar height
                        )))


    # Hexagonal Brillouin Zone vertices
    BZ_vertices = np.array([
        0.666 * b1 - 0.333 * b2, 
        0.333 * b1 + 0.333 * b2, 
        -0.333 * b1 + 0.666 * b2, 
        -0.666 * b1 + 0.333 * b2, 
        -0.333 * b1 - 0.333 * b2, 
        0.333 * b1 - 0.666 * b2,
        0.666 * b1 - 0.333 * b2])

    # High-symmetry points in units of (2π/a)
    Gamma = np.array([0, 0])
    M = 0.5 * b2
    K = -0.3333333333 * b1 + 0.6666666667 * b2


    # Add arrows for b1 and b2
    fig.add_annotation(
        x=b1[0], y=b1[1],
        ax=0, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=2,
        arrowcolor="green",
        text="b1",
        font=dict(size=12, color="green"),
        yshift=0
    )

    fig.add_annotation(
        x=b2[0], y=b2[1],
        ax=0, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=2,
        arrowcolor="purple",
        text="b2",
        font=dict(size=12, color="purple"),
        yshift=0
    )

    # Path: Γ → M → K → Γ
    path = np.array([Gamma, M, K, Gamma])

    # Plot high-symmetry points
    high_symmetry_labels = ['Γ', 'M', 'K']
    high_symmetry_points = [Gamma, M, K]
    for point, label in zip(high_symmetry_points, high_symmetry_labels):
        fig.add_trace(go.Scatter(
            x=[point[0]], 
            y=[point[1]], 
            mode='markers+text',
            text=[label],
            textposition="top center",
            marker=dict(color='red', size=10),
            name=label,
            showlegend=False
        ))
        
    # Plot path
    fig.add_trace(go.Scatter(
        x=path[:, 0], 
        y=path[:, 1], 
        mode='lines+markers', 
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(color='red', size=6),
        name='Path: Γ → M → K → Γ'
    ))


    # Plot BZ
    fig.add_trace(go.Scatter(
        x=BZ_vertices[:, 0], 
        y=BZ_vertices[:, 1], 
        mode='lines', 
        line=dict(color='black', width=2),
        showlegend=False
    ))


    fig.update_layout(
        autosize=False,
        width=800,  # Width of the figure
        height=800,  # Height of the figure
            xaxis=dict(
            scaleanchor="y",  # Match the scale of the x-axis with the y-axis
            title="kx cart in 2 pi / alat",
            range=[-1, 1]
        ),
        yaxis=dict(title="ky cart in 2 pi / alat", range=[-1, 1]),
        title="J(q)"
    )
    fig.show()



# def plot_JqPath():
#     fig, (dd) = plt.subplots()#1, 2, gridspec_kw={'width_ratios': [1.5, 1]}

#     normal_ticks = calc.HighSymPointsDists/qe2wan
#     label_ticks = calc.HighSymPointsNames


#     dd.plot(kpath_draw,#/1.27733*0.5, 
#             (Jqs00 - Jself)*2/(s**2)*1000,  color='red', linewidth=0.7,
#                 alpha=1, marker=">",  markersize=3.0,  label=r"Re[$J_q'$]")

#     # dd.plot(kpath_draw,#/1.27733*0.5, 
#     #         (Jqs11 - Jself)*2/(1.5**2)*1000, '--',  color='red', linewidth=0.7,
#     #             alpha=1, marker=">",  markersize=3.0,  label=r"Re[$J_q'$]")


#     dd.plot(kpath_draw,#/1.27733*0.5, 
#             (np.real(Jqs01) )*2/(s**2)*1000,  color='red', linewidth=0.7,
#                 alpha=1, marker="D",  markersize=3.0,  label=r"Re[$J_{01}(q)'$]")

#     dd.plot(kpath_draw,#/1.27733*0.5, 
#             (np.imag(Jqs01) )*2/(s**2)*1000,  color='blue', linewidth=0.7,
#                 alpha=1, marker="o",  markersize=3.0,  label=r"Im[$J_{01}(q)'$]")


#     # dd.plot(kpath_draw,#/1.27733*0.5, 
#     #         (np.real(Jqs01) )*2/(1.5**2)*1000,  color='red', linewidth=0.7,
#     #             alpha=1, marker="D",  markersize=3.0,  label=r"Re[$J_{10}(q)'$]")

#     # dd.plot(kpath_draw,#/1.27733*0.5, 
#     #         (np.imag(Jqs01) )*2/(1.5**2)*1000,  color='blue', linewidth=0.7,
#     #             alpha=1, marker="o",  markersize=3.0,  label=r"Im[$J_{10}(q)'$]")



#     dd.set_ylabel('J(q) (meV)')  # Add an x-label to the axes.
#     dd.set_xlabel('q')  # Add a y-label to the axes.
#     dd.set_title(f'J(q) beta={beta:.0f}')
#     dd.legend(prop={'size': 9}, frameon=False, loc='upper right', bbox_to_anchor=(0.9, 0.99))  # Add a legend.
#     # locator = AutoMinorLocator()
#     # dd.yaxis.set_minor_locator(MultipleLocator(0.05))

#     dd.set_xticks(normal_ticks, label_ticks)
#     dd.grid(axis='x')
#     dd.xaxis.set_minor_locator(AutoMinorLocator())
#     dd.tick_params(top=True, right=True, which='minor',length=2, width=0.2, direction="in")
#     dd.tick_params(top=True, right=True, which='major',length=3.5, width=0.4, labelsize=8, direction="in")

#     dd.set_xlim(normal_ticks[0], normal_ticks[-1])
#     # dd.set_ylim(-0.3, 0.2)
#     dd.text(-0.1, 1.0, 'a)', transform=dd.transAxes,
#             fontsize=10, fontweight='normal', va='top', ha='right')


#     # dd.yaxis.set_major_locator(MultipleLocator(0.1))

#     plt.rcParams['axes.linewidth'] = 0.3
#     width = 5
#     fig.set_figwidth(6)     #  ширина в дюймах (2,54)
#     fig.set_figheight(5/1.6)    #  высота в дюймах (2,54)
#     fig.tight_layout()
#     # plt.savefig('./2pub/pics/Jq_beta_10.eps', 
#     #             format='eps', dpi=200, bbox_inches='tight')

#     plt.show()


# def plotJ_Real(Js_neib00, Js_neib01, sorted_vert_J, J_path_plt):

#     def print_stats(Js):
#         tmp = 0
#         total = 0
#         plot_data = []
#         print(f'\t #Neib   \t Sum J (meV)  \t J mean (meV)')
#         print('-'*50)
#         for order, order_dist in enumerate(sorted_vert_J.keys()):
#             num_vert = len(sorted_vert_J[order_dist])
#             J_params = Js[tmp:tmp + num_vert]*1000
#             tmp = tmp + num_vert
#             total += np.real(np.sum(J_params))
#             print(f' \
#                 {num_vert}\t \
#                 {np.real(np.sum(J_params)):.2f}\t \
#                 {np.real(np.sum(J_params)/num_vert):.2f} \
#                 ')
#             plot_data.append([order+1, np.real(np.sum(J_params)/num_vert)])
#         plot_data = np.array(plot_data)

#         print(f'Total {np.real(total):.2f} (meV) ' + 'FM' if np.real(total) > 0 else 'AFM')
#         return plot_data

#     print('\nJ00')
#     plot_data00 = print_stats(Js_neib00)
#     print('\nJ01')
#     plot_data01 = print_stats(Js_neib01)


#     fig, dd = plt.subplots()

#     dd.plot(plot_data00[:, 0], plot_data00[:,1])
#     dd.scatter(plot_data00[:, 0], plot_data00[:,1],color='black', 
#                 alpha=1, marker="D",  s=15.0, label=r'Re[$J_{00}(r)$]')

#     dd.plot(plot_data01[:, 0], plot_data01[:,1])
#     dd.scatter(plot_data01[:, 0], plot_data01[:,1],color='red', 
#                 alpha=1, marker="D",  s=15.0, label=r'Re[$J_{01}(r)$]')


#     dd.set_ylabel('E (meV)')  # Add an x-label to the axes.
#     dd.set_xlabel('Neighbour order')  # Add a y-label to the axes.
#     # dd.legend(prop={'size': 9}, frameon=False)  # Add a legend.

#     dd.xaxis.set_major_locator(MultipleLocator(1))
#     dd.yaxis.set_minor_locator(MultipleLocator(0.5))
#     dd.tick_params(top=False, right=False, which='minor',length=2, width=0.2, direction="in")
#     dd.tick_params(top=False, right=False, which='major',length=3.5, width=0.4, labelsize=8, direction="in")
#     dd.hlines(0, xmin=0, xmax=max(J_path_plt), colors='black', linewidth=0.4)
#     dd.set_xlim(0.01, J_path_plt[-1])

#     plt.rcParams['axes.linewidth'] = 0.5
#     width = 5
#     fig.set_figwidth(5)     #  ширина в дюймах (2,54)
#     fig.set_figheight(5/1.6)    #  высота в дюймах (2,54)
#     fig.tight_layout()
#     # plt.savefig('./2pub/pics/J_r_beta_10.eps', 
#     #             format='eps', dpi=200, bbox_inches='tight')

#     plt.show()




def get_brillouin_zone_3d(cell):
    """
    Uses the k-space vectors and voronoi analysis to define
    the BZ of the system

    Args:
        cell: a 3x3 matrix defining the basis vectors in
        reciprocal space

    Returns:
        vor.vertices[bz_vertices]: vertices of BZ
        bz_ridges: edges of the BZ
        bz_facets: BZ facets

    """

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    from scipy.spatial import Voronoi

    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):

        if pid[0] == 13 or pid[1] == 13:
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))

    return vor.vertices[bz_vertices], bz_ridges, bz_facets


# vertices, ridges, _ = get_brillouin_zone_3d(calc.bcell)