import json
import os, random, string, time, sys, signal
from pathlib import Path
from pprint import pprint

import numpy as np

from colabdesign.rf.utils import fix_contigs, fix_partial_contigs, fix_pdb, sym_it
from colabdesign.shared.protein import pdb_to_string

CWD = Path(__file__).parent

sys.path.append(CWD.resolve())

print(f"Set working directory: {CWD}")

from rfdiffusion.inference.utils import parse_pdb


def get_pdb(pdb_code):
    if os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        if not os.path.isfile(f"{pdb_code}.pdb1"):
            os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.pdb1.gz")
            os.system(f"gunzip {pdb_code}.pdb1.gz")
        return f"{pdb_code}.pdb1"
    else:
        os.system(
            f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb"
        )
        return f"AF-{pdb_code}-F1-model_v3.pdb"


def run_ananas(pdb_str, path, sym=None):
    pdb_filename = f"outputs/{path}/ananas_input.pdb"
    out_filename = f"outputs/{path}/ananas.json"
    with open(pdb_filename, "w") as handle:
        handle.write(pdb_str)

    cmd = f"./ananas {pdb_filename} -u -j {out_filename}"
    if sym is None:
        os.system(cmd)
    else:
        os.system(f"{cmd} {sym}")

    # parse results
    try:
        out = json.loads(open(out_filename, "r").read())
        results, AU = out[0], out[-1]["AU"]
        group = AU["group"]
        chains = AU["chain names"]
        rmsd = results["Average_RMSD"]
        print(f"AnAnaS detected {group} symmetry at RMSD:{rmsd:.3}")

        C = np.array(results["transforms"][0]["CENTER"])
        A = [np.array(t["AXIS"]) for t in results["transforms"]]

        # apply symmetry and filter to the asymmetric unit
        new_lines = []
        for line in pdb_str.split("\n"):
            if line.startswith("ATOM"):
                chain = line[21:22]
                if chain in chains:
                    x = np.array([float(line[i : (i + 8)]) for i in [30, 38, 46]])
                    if group[0] == "c":
                        x = sym_it(x, C, A[0])
                    if group[0] == "d":
                        x = sym_it(x, C, A[1], A[0])
                    coord_str = "".join(["{:8.3f}".format(a) for a in x])
                    new_lines.append(line[:30] + coord_str + line[54:])
            else:
                new_lines.append(line)
        return results, "\n".join(new_lines)

    except:
        return None, pdb_str


def run(command, steps, num_designs=1, visual="none"):
    def run_command_and_get_pid(command):
        pid_file = "/dev/shm/pid"
        os.system(f"nohup {command} > /dev/null & echo $! > {pid_file}")
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        os.remove(pid_file)
        return pid

    def is_process_running(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    # run_output = widgets.Output()
    # progress = widgets.FloatProgress(
    #     min=0, max=1, description="running", bar_style="info"
    # )
    # display(widgets.VBox([progress, run_output]))

    # clear previous run
    for n in range(steps):
        if os.path.isfile(f"/dev/shm/{n}.pdb"):
            os.remove(f"/dev/shm/{n}.pdb")

    pid = run_command_and_get_pid(command)
    try:
        fail = False
        for _ in range(num_designs):
            # for each step check if output generated
            for n in range(steps):
                wait = True
                while wait and not fail:
                    time.sleep(0.1)
                    if os.path.isfile(f"/dev/shm/{n}.pdb"):
                        pdb_str = open(f"/dev/shm/{n}.pdb").read()
                        if pdb_str[-3:] == "TER":
                            wait = False
                        elif not is_process_running(pid):
                            fail = True
                    elif not is_process_running(pid):
                        fail = True

                if fail:
                    print(f"Failed on step: {n}")
                    # progress.bar_style = "danger"
                    # progress.description = "failed"
                    break

                else:
                    pass
                    # progress.value = (n + 1) / steps
                    # if visual != "none":
                    #     with run_output:
                    #         run_output.clear_output(wait=True)
                    #         if visual == "image":
                    #             xyz, bfact = get_ca(f"/dev/shm/{n}.pdb", get_bfact=True)
                    #             fig = plt.figure()
                    #             fig.set_dpi(100)
                    #             fig.set_figwidth(6)
                    #             fig.set_figheight(6)
                    #             ax1 = fig.add_subplot(111)
                    #             ax1.set_xticks([])
                    #             ax1.set_yticks([])
                    #             plot_pseudo_3D(xyz, c=bfact, cmin=0.5, cmax=0.9, ax=ax1)
                    #             plt.show()
                    #         if visual == "interactive":
                    #             view = py3Dmol.view(
                    #                 js="https://3dmol.org/build/3Dmol.js"
                    #             )
                    #             view.addModel(pdb_str, "pdb")
                    #             view.setStyle(
                    #                 {
                    #                     "cartoon": {
                    #                         "colorscheme": {
                    #                             "prop": "b",
                    #                             "gradient": "roygb",
                    #                             "min": 0.5,
                    #                             "max": 0.9,
                    #                         }
                    #                     }
                    #                 }
                    #             )
                    #             view.zoomTo()
                    #             view.show()
                if os.path.exists(f"/dev/shm/{n}.pdb"):
                    os.remove(f"/dev/shm/{n}.pdb")
            if fail:
                print(f"Failed: 105")
                # progress.bar_style = "danger"
                # progress.description = "failed"
                break

        while is_process_running(pid):
            time.sleep(0.1)

    except KeyboardInterrupt:
        os.kill(pid, signal.SIGTERM)
        # progress.bar_style = "danger"
        # progress.description = "stopped"


def run_diffusion(
    contigs,
    path,
    pdb=None,
    iterations=50,
    symmetry="none",
    order=1,
    hotspot=None,
    chains=None,
    add_potential=False,
    num_designs=1,
    visual="none",
):
    full_path = f"outputs/{path}"
    os.makedirs(full_path, exist_ok=True)
    opts = [
        f"inference.output_prefix={full_path}",
        f"inference.num_designs={num_designs}",
    ]

    if chains == "":
        chains = None

    # determine symmetry type
    if symmetry in ["auto", "cyclic", "dihedral"]:
        if symmetry == "auto":
            sym, copies = None, 1
        else:
            sym, copies = {
                "cyclic": (f"c{order}", order),
                "dihedral": (f"d{order}", order * 2),
            }[symmetry]
    else:
        symmetry = None
        sym, copies = None, 1

    # determine mode
    contigs = contigs.replace(",", " ").replace(":", " ").split()
    is_fixed, is_free = False, False
    fixed_chains = []
    for contig in contigs:
        for x in contig.split("/"):
            a = x.split("-")[0]
            if a[0].isalpha():
                is_fixed = True
                if a[0] not in fixed_chains:
                    fixed_chains.append(a[0])
            if a.isnumeric():
                is_free = True
    if len(contigs) == 0 or not is_free:
        mode = "partial"
    elif is_fixed:
        mode = "fixed"
    else:
        mode = "free"

    # fix input contigs
    if mode in ["partial", "fixed"]:
        pdb_str = pdb_to_string(get_pdb(pdb), chains=chains)
        if symmetry == "auto":
            a, pdb_str = run_ananas(pdb_str, path)
            if a is None:
                print(f"ERROR: no symmetry detected")
                symmetry = None
                sym, copies = None, 1
            else:
                if a["group"][0] == "c":
                    symmetry = "cyclic"
                    sym, copies = a["group"], int(a["group"][1:])
                elif a["group"][0] == "d":
                    symmetry = "dihedral"
                    sym, copies = a["group"], 2 * int(a["group"][1:])
                else:
                    print(
                        f'ERROR: the detected symmetry ({a["group"]}) not currently supported'
                    )
                    symmetry = None
                    sym, copies = None, 1

        elif mode == "fixed":
            pdb_str = pdb_to_string(pdb_str, chains=fixed_chains)

        pdb_filename = f"{full_path}/input.pdb"
        with open(pdb_filename, "w") as handle:
            handle.write(pdb_str)

        parsed_pdb = parse_pdb(pdb_filename)
        opts.append(f"inference.input_pdb={pdb_filename}")
        if mode in ["partial"]:
            iterations = int(80 * (iterations / 200))
            opts.append(f"diffuser.partial_T={iterations}")
            contigs = fix_partial_contigs(contigs, parsed_pdb)
        else:
            opts.append(f"diffuser.T={iterations}")
            contigs = fix_contigs(contigs, parsed_pdb)
    else:
        opts.append(f"diffuser.T={iterations}")
        parsed_pdb = None
        contigs = fix_contigs(contigs, parsed_pdb)

    if hotspot is not None and hotspot != "":
        opts.append(f"ppi.hotspot_res=[{hotspot}]")

    # setup symmetry
    if sym is not None:
        sym_opts = ["--config-name symmetry", f"inference.symmetry={sym}"]
        if add_potential:
            sym_opts += [
                "'potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"]'",
                "potentials.olig_intra_all=True",
                "potentials.olig_inter_all=True",
                "potentials.guide_scale=2",
                "potentials.guide_decay=quadratic",
            ]
        opts = sym_opts + opts
        contigs = sum([contigs] * copies, [])

    opts.append(f"'contigmap.contigs=[{' '.join(contigs)}]'")
    opts += ["inference.dump_pdb=True", "inference.dump_pdb_path='/dev/shm'"]

    print("mode:", mode)
    print("output:", full_path)
    print("contigs:", contigs)

    opts_str = " ".join(opts)
    cmd = f"./scripts/run_inference.py {opts_str}"
    print(cmd)

    # RUN
    run(cmd, iterations, num_designs, visual=visual)

    # fix pdbs
    for n in range(num_designs):
        pdbs = [
            f"outputs/traj/{path}_{n}_pX0_traj.pdb",
            f"outputs/traj/{path}_{n}_Xt-1_traj.pdb",
            f"{full_path}_{n}.pdb",
        ]
        for pdb in pdbs:
            with open(pdb, "r") as handle:
                pdb_str = handle.read()
            with open(pdb, "w") as handle:
                handle.write(fix_pdb(pdb_str, contigs))

    return contigs, copies


def main():
    # @title run **RFdiffusion** to generate a backbone
    name = "petase"  # @param {type:"string"}
    contigs = "A1-100"  # @param {type:"string"}
    pdb = "./6ij6.pdb"  # @param {type:"string"}
    iterations = 50  # @param ["25", "50", "100", "150", "200"] {type:"raw"}
    hotspot = ""  # @param {type:"string"}
    num_designs = 1  # @param ["1", "2", "4", "8", "16", "32"] {type:"raw"}
    visual = "image"  # @param ["none", "image", "interactive"]
    # @markdown ---
    # @markdown **symmetry** settings
    # @markdown ---
    symmetry = "none"  # @param ["none", "auto", "cyclic", "dihedral"]
    order = 1  # @param ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"] {type:"raw"}
    chains = "A"  # @param {type:"string"}
    add_potential = True  # @param {type:"boolean"}
    # @markdown - `symmetry='auto'` enables automatic symmetry dectection with [AnAnaS](https://team.inria.fr/nano-d/software/ananas/).
    # @markdown - `chains="A,B"` filter PDB input to these chains (may help auto-symm detector)
    # @markdown - `add_potential` to discourage clashes between chains

    # determine where to save
    path = name
    while os.path.exists(f"outputs/{path}_0.pdb"):
        path = (
            name
            + "_"
            + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        )

    flags = {
        "contigs": contigs,
        "pdb": pdb,
        "order": order,
        "iterations": iterations,
        "symmetry": symmetry,
        "hotspot": hotspot,
        "path": path,
        "chains": chains,
        "add_potential": add_potential,
        "num_designs": num_designs,
        "visual": visual,
    }

    for k, v in flags.items():
        if isinstance(v, str):
            flags[k] = v.replace("'", "").replace('"', "")

    pprint(flags)

    run_diffusion(**flags)


if __name__ == "__main__":
    main()
