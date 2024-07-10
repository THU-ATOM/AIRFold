import argparse
import sys
import os
from os.path import isfile, isdir, join
import numpy as np
import multiprocessing
import torch
from lib.tool.gcpl.modules.QA_utils.folding import process_model
import warnings
warnings.filterwarnings("ignore")


def main():

    parser = argparse.ArgumentParser(description="predictor network error")
    parser.add_argument("input",
                        action="store",
                        help="path to input")

    parser.add_argument("s_emb",
                        action="store",
                        help="path to a3m")

    parser.add_argument("output",
                        action="store", nargs=argparse.REMAINDER,
                        help="path to output")
    
    parser.add_argument("--pdb",
                        "-pdb",
                        action="store_true",
                        default=False,
                        help="Running on a single pdb ")
    
    parser.add_argument("--csv",
                        "-csv",
                        action="store_true",
                        default=False,
                        help="Writing results to a csv file ")

    parser.add_argument("--per_res_only",
                        "-pr",
                        action="store_true",
                        default=False,
                        help="Store per-residue accuracy only")
    
    parser.add_argument("--leaveTempFile",
                        "-lt",
                        action="store_true",
                        default=False,
                        help="Leaving temporary files")
    
    parser.add_argument("--process",
                        "-p", action="store",
                        type=int,
                        default=1,
                        help="Specifying # of cpus to use for featurization")
    
    parser.add_argument("--featurize",
                        "-f",
                        action="store_true",
                        default=False,
                        help="Running only the featurization part")
    
    parser.add_argument("--reprocess",
                        "-r", action="store_true",
                        default=False,
                        help="Reprocessing all feature files")
    
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="Activating verbose flag ")

    parser.add_argument("--ensemble",
                        "-e", 
                        action="store_true",
                        default=False,
                        help="Running with ensembling of 4 models. ")



    args = parser.parse_args()
    csvfilename = "result.csv"

    if len(args.output)>1:
        print(f"Only one output folder can be specified, but got {args.output}", file=sys.stderr)
        return -1
    
    if len(args.output)==0:
        args.output = ""
    else:
        args.output = args.output[0]

    if args.input.endswith('.pdb'):
        args.pdb = True
    
    if args.output.endswith(".csv"):
        args.csv = True

    if not args.pdb:
        if not isdir(args.input):
            print("Input folder does not exist.", file=sys.stderr)
            return -1

        if args.output == "":
            args.output = args.input
        else:
            if not args.csv and not isdir(args.output):
                if args.verbose: print("Creating output folder:", args.output)
                os.mkdir(args.output)
            
            # if csv, do it in place.
            elif args.csv:
                csvfilename = args.output
                args.output = args.input

    else:
        if not isfile(args.input):
            print("Input file does not exist.", file=sys.stderr)
            return -1

        if args.output == "":
            args.output = os.path.splitext(args.input)[0]+".npz"

        if not(".pdb" in args.input and ".npz" in args.output):
            print("Input needs to be in .pdb format, and output needs to be in .npz format.", file=sys.stderr)
            return -1
        
    # script_dir = os.path.dirname(__file__)[:-3]
    modelpath = "/data/protein/datasets_2024/GraphCPLMQA/QA_Model"

    if not isdir(modelpath):
        print("Model checkpoint does not exist", file=sys.stderr)
        return -1

    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import lib.tool.gcpl.modules as qa_file

    num_process = 1
    if args.process > 1:
        num_process = args.process
        

    if not args.pdb:
        samples = [i[:-4] for i in os.listdir(args.input) if isfile(args.input+"/"+i) and i[-4:] == ".pdb" and i[0]!="."]
        ignored = [i[:-4] for i in os.listdir(args.input) if not(isfile(args.input+"/"+i) and i[-4:] == ".pdb" and i[0]!=".")]
        if args.verbose: 
            print("# samples:", len(samples))
            if len(ignored) > 0:
                print("# files ignored:", len(ignored))


        inputs = [join(args.input, s)+".pdb" for s in samples]
        tmpoutputs = [join(args.output, s)+".features.npz" for s in samples]
        
        if not args.reprocess:
            arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if not isfile(tmpoutputs[i])]
            already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
            if args.verbose: 
                print("Featurizing", len(arguments), "samples.", len(already_processed), "are already processed.")
        else:
            arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs))]
            already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
            if args.verbose: 
                print("Featurizing", len(arguments), "samples.", len(already_processed), "are re-processed.")

        if num_process == 1:
            for a in arguments:
                qa_file.process(a)
        else:
            pool = multiprocessing.Pool(num_process)
            out = pool.map(qa_file.process, arguments)
            

        if args.featurize:
            return 0
        
        if args.verbose: print("using", modelpath)

        
        samples = [s for s in samples if isfile(join(args.output, s+".features.npz"))]

        if args.ensemble:
            modelnames = ["best.pkl", "second.pkl", "third.pkl", "fourth.pkl"]
        else:
            modelnames = ["GCPL.pkl"]
        
        result = {}

        for modelname in modelnames:
            model = qa_file.QA(num_channel=128)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("device",device)
            checkpoint = torch.load(join(modelpath, modelname), map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            for s in samples:
                # try:
                with torch.no_grad():
                    if args.verbose: print("Predicting for", s)
                    out_path=join(args.output, s + ".node.2e1S.npz")
                    if os.path.exists(out_path):
                        print(args.output, s + ".node.2e1S.npz is already exists.")
                        continue
                    filename = join(args.output, s+".features.npz")
                    bertname = ""

                    path_p_coor = args.input + "/" + s + ".pdb"
                    model_coords, _ = process_model(path_p_coor)

                    (idx, val), (f1d, bert), f2d, _ = qa_file.getData(filename, model_coords, bertpath = bertname)

                    
                    semb_path = args.s_emb + "/" + s + ".emb.if.npz"
                    stru_emb = np.load(semb_path)["rep"]
                    f1d = np.concatenate([f1d, stru_emb], axis=-1)

                    emb_path = join(args.output, "seq_emb/" + s + ".2e1.npz")
                    msa_emb = np.load(emb_path)
                    node_emb = np.expand_dims(msa_emb["only_last"],0)

                    f1d = torch.Tensor(f1d).to(device)
                    f2d = torch.Tensor(np.expand_dims(f2d.transpose(2, 0, 1), 0)).to(device)
                    idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
                    val = torch.Tensor(val).to(device)
                    node_emb = torch.Tensor(node_emb).to(device)


                    output, mask_logits, deviation_logits = model(idx, val, f1d, f2d, node_emb, model_coords.to(device))


                    coords_loss = output.coords_loss
                    bondlen_loss = output.bondlen_loss
                    p_lddt_loss = output.p_lddt_loss

                    lddt = output.p_lddt_pred
                    ipa_coords = output.coords

                    t = result.get(s, [])
                    t.append(np.mean(lddt.cpu().detach().numpy()))
                    result[s] = t

                    if not args.csv:
                        if args.ensemble:
                            if args.per_res_only:
                                np.savez_compressed(join(args.output, s+"_"+modelname[:-4]+".node.2e1S.npz"),
                                                    lddt = lddt.cpu().detach().numpy().astype(np.float16),

                                                    )
                            else:
                                np.savez_compressed(join(args.output, s+"_"+modelname[:-4]+".node.2e1S.npz"),
                                                    lddt = lddt.cpu().detach().numpy().astype(np.float16),
                                                    ipa_coords=ipa_coords.cpu().detach().numpy().astype(np.float16),
                                                    )
                        else:
                            if args.per_res_only:
                                np.savez_compressed(join(args.output, s+".node.2e1S.npz"),
                                                    lddt = lddt.cpu().detach().numpy().astype(np.float16),
                                                    ipa_coords=ipa_coords.cpu().detach().numpy().astype(np.float16),
                                                    )
                            else:
                                np.savez_compressed(join(args.output, s+".node.2e1S.npz"),
                                                    lddt = lddt.cpu().detach().numpy().astype(np.float16),
                                                    ipa_coords=ipa_coords.cpu().detach().numpy().astype(np.float16),
                                                    )
                    print("success for ",s+"_"+modelname[:-4])
                # except:
                #     print("Failed to predict for", join(args.output, s+"_"+modelname[:-4]+".npz"))
                #     current_dir=os.path.abspath(os.path.dirname(__file__))
                #     with open(current_dir+"/predict_gen_problem.txt","a") as f:
                #         f.write("Failed to predict for"+join(args.output, s+"_"+modelname[:-4]+".node.2e1S.npz")+"\n")
                #     continue

        if not args.csv:
            if args.ensemble:
                qa_file.merge(samples, args.output, verbose=args.verbose)
            
            if not args.leaveTempFile:
                qa_file.clean(samples,
                          args.output,
                          verbose=args.verbose,
                          ensemble=args.ensemble)
        else:
            # Take average of outputs
            csvfile = open(csvfilename, "w")
            csvfile.write("sample\tcb-lddt\n")
            for s in samples:
                line = "%s\t%.4f\n"%(s, np.mean(result[s]))
                csvfile.write(line)
            csvfile.close()
            
    # Processing for single sample
    else:
        infilepath = args.input
        outfilepath = args.output
        infolder = "/".join(infilepath.split("/")[:-1])
        insamplename = infilepath.split("/")[-1][:-4]
        outfolder = "/".join(outfilepath.split("/")[:-1])
        outsamplename = outfilepath.split("/")[-1][:-4]
        feature_file_name = join(outfolder, outsamplename+".features.npz")
        if args.verbose: 
            print("only working on a file:", outfolder, outsamplename)
        # Process if file does not exists or reprocess flag is set
        
        if (not isfile(feature_file_name)) or args.reprocess:
            qa_file.process((join(infolder, insamplename+".pdb"),
                                feature_file_name,
                                args.verbose))
            
        if isfile(feature_file_name):

            model = qa_file.QA()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")
            model.load_state_dict(torch.load(join(modelpath, "best.pkl"), map_location=device)['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Actual prediction
            with torch.no_grad():
                if args.verbose: print("Predicting for", outsamplename) 
                (idx, val), (f1d, bert), f2d, dmy = qa_file.getData(feature_file_name)
                f1d = torch.Tensor(f1d).to(device)
                f2d = torch.Tensor(np.expand_dims(f2d.transpose(2,0,1), 0)).to(device)
                idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
                val = torch.Tensor(val).to(device)

                deviation, mask, lddt, lddt_direct, dmy = model(idx, val, f1d, f2d)
                #deviation, mask, lddt, dmy = model(f1d, f2d)
                if args.per_res_only:
                    np.savez_compressed(outsamplename+".node.2e1S.npz",
                            lddt = lddt.cpu().detach().numpy().astype(np.float16),
                            lddt_direct = lddt_direct.cpu().detach().numpy().astype(np.float16))
                else:
                    np.savez_compressed(outsamplename+".node.2e1S.npz",
                            lddt = lddt.cpu().detach().numpy().astype(np.float16),
                            lddt_direct=lddt_direct.cpu().detach().numpy().astype(np.float16),
                            deviation = deviation.cpu().detach().numpy().astype(np.float16),
                            mask = mask.cpu().detach().numpy().astype(np.float16))

            if not args.leaveTempFile:
                qa_file.clean([outsamplename],
                          outfolder,
                          verbose=args.verbose,
                          ensemble=False)
        else:
            print(f"Feature file does not exist: {feature_file_name}", file=sys.stderr)

            
if __name__== "__main__":
    main()
