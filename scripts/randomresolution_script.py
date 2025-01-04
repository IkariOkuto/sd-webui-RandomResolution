import math

import gradio as gr
import modules.scripts as scripts
from modules import deepbooru, images, processing, shared
from modules.processing import Processed
from modules.shared import opts, state
import random
import os

class Script(scripts.Script):
    RR_MAXSIZE = 6 #Size of Resolution Type.
    RR_FIRSTOPEN = 2

    def title(self):
        return "Random Resolution"

    def ui(self, is_img2img):
        widthlist = []
        heightlist = []
        weightlist = []
        accordionlist = []
        numlist = []
        initwidthtext = "768"
        initheighttext = "768"
        initweighttext = "1"

        presetslist = ["Preset1", "Preset2", "Preset3", "Preset4", "Preset5", "Preset6"]

        # 添え字用.
        for i in range(Script.RR_MAXSIZE):
            grnum = gr.Number(value=i, visible=False, precision=0)
            numlist.append(grnum)
        # 表示数.
        viewnum = gr.Number(value=Script.RR_FIRSTOPEN, visible=False, precision=0)

        for i in range(Script.RR_MAXSIZE - 1):
            initwidthtext += ",768"
            initheighttext += ",768"
            initweighttext += ",1"

        # Prests
        dirpath = os.path.join(scripts.basedir(), "extensions", "sd-webui-random-resolution")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = os.path.join(dirpath, "presets.txt")
        if not os.path.exists(filepath):
            f = open(filepath, 'w')
            f.writelines(initwidthtext)
            f.writelines(initheighttext)
            f.writelines(initweighttext)
            f.close()


        #UI
        with gr.Column():
            with gr.Row():
                 dropdown = gr.Dropdown(choices=presetslist, value="Preset1", label="Presets", scale=2)
                 load_btn = gr.Button("Load", scale=1)
                 save_btn = gr.Button("Save", scale=1)
            with gr.Row():
                 new_btn = gr.Button("New Val")
                 del_btn = gr.Button("Del Val")
            # 出力用.
            widths = gr.HTML(label='Width (comma separated)', value=initwidthtext, visible=False)
            heights = gr.HTML(label='Height (comma separated)', value=initheighttext, visible=False)
            weights = gr.HTML(label='Weight (comma separated)', value=initweighttext, visible=False)

            
            for i in range(Script.RR_MAXSIZE):
                labeltmp= "Resolution " + str(i+1)
                if i < Script.RR_FIRSTOPEN:
                    with gr.Accordion(label=labeltmp,open=True,visible=True) as acc:
                        accordionlist.append(acc)
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2):
                                width = gr.Slider(minimum=64, maximum=4096, value=768, step=8, label="Width_"+ str(i+1))
                                height = gr.Slider(minimum=64, maximum=4096, value=768, step=8, label="Height_"+ str(i+1))
                                widthlist.append(width)
                                heightlist.append(height)
                            with gr.Column(scale=1):
                                weight = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Weight_"+ str(i+1))
                                weightlist.append(weight)
                else:
                    with gr.Accordion(label=labeltmp,open=True,visible=False) as acc:
                        accordionlist.append(acc)
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2):
                                width = gr.Slider(minimum=64, maximum=4096, value=768, step=8, label="Width_"+ str(i+1))
                                height = gr.Slider(minimum=64, maximum=4096, value=768, step=8, label="Height_"+ str(i+1))
                                widthlist.append(width)
                                heightlist.append(height)
                            with gr.Column(scale=1):
                                weight = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Weight_"+ str(i+1))
                                weightlist.append(weight)


            #accordionlist[0].update(visible=True)
        
        # Event
        def checkgroup(viewnum):
            funclist = [gr.Number.update(value=viewnum)]
            for i in range(Script.RR_MAXSIZE):
                if i < viewnum:
                    funclist.append(gr.Accordion.update(visible=True))
                else:
                    funclist.append(gr.Accordion.update(visible=False))
            return tuple(funclist)

        def addgroup(viewnum):
            if viewnum < Script.RR_MAXSIZE:
                viewnum += 1
            return checkgroup(viewnum)
        
        def delgroup(viewnum):
            if viewnum > 1:
                viewnum -= 1
            return checkgroup(viewnum)

        changelist = [viewnum] + accordionlist


        new_btn.click(
            addgroup,
            [viewnum],
            changelist,
        )

        del_btn.click(
            delgroup,
            [viewnum],
            changelist,
        )

        def updatetext(id, slider, text, *args):
            l = text.split(',')
            l[id] = str(slider)
            tmptext = ','.join(l)
            return gr.HTML.update(value=tmptext)

        # すべてのwidthスライダーに更新処理を設定.
        for i, w in enumerate(widthlist):
            w.release(updatetext, [numlist[i], w, widths], [widths],)
        for i, h in enumerate(heightlist):
            h.release(updatetext, [numlist[i], h, heights], [heights],)
        for i, w in enumerate(weightlist):
            w.release(updatetext, [numlist[i], w, weights], [weights],)

        def nochangeall():
            #[viewnum] + accordionlist +  [widths, heights, weights] + widthlist + heightlist + weightlist
            nochange = [gr.Number().update()]
            for i in range(Script.RR_MAXSIZE):
                nochange.append(gr.Accordion.update())

            nochange.append(gr.HTML().update()) #w
            nochange.append(gr.HTML().update()) #h
            nochange.append(gr.HTML().update()) #w

            for i in range(Script.RR_MAXSIZE):
                nochange.append(gr.Slider().update()) #w
                nochange.append(gr.Slider().update()) #h
                nochange.append(gr.Slider().update()) #w

            return tuple(nochange)


        def loadpreset(dropdown, widths, heights, weights):
            
            print("hogehogehogehoge")
            dirpath = os.path.join(scripts.basedir(), "extensions", "sd-webui-random-resolution")
            if not os.path.exists(dirpath):
                return nochangeall()

            filepath = os.path.join(dirpath, dropdown + ".txt")
            if not os.path.exists(filepath):
                return nochangeall()

            f = open(filepath, 'r')
            lines = f.readlines()

            funclist= list(checkgroup(int(lines[0])))

            o_widl = widths.split(',')
            o_heil = heights.split(',')
            o_weil = weights.split(',')

            p_widl = lines[1].replace('\n','').split(',')
            p_heil = lines[2].replace('\n','').split(',')
            p_weil = lines[3].replace('\n','').split(',')

            loopsize = min(len(o_widl), len(p_widl))
            for i in range(loopsize):
                o_widl[i] = p_widl[i]
                o_heil[i] = p_heil[i]
                o_weil[i] = p_weil[i]

            funclist.append(gr.HTML.update(value=(','.join(o_widl)))) #w
            funclist.append(gr.HTML.update(value=(','.join(o_heil))))#h
            funclist.append(gr.HTML.update(value=(','.join(o_weil)))) #w


            for i in range(Script.RR_MAXSIZE):
                 funclist.append(gr.Slider.update(value=int(o_widl[i])))

            for i in range(Script.RR_MAXSIZE):
                 funclist.append(gr.Slider.update(value=int(o_heil[i])))

            for i in range(Script.RR_MAXSIZE):
                 funclist.append(gr.Slider.update(value=int(o_weil[i])))

            return tuple(funclist)

        def savepreset(dropdown, viewnum, widths, heights, weights):
            dirpath = os.path.join(scripts.basedir(), "extensions", "sd-webui-random-resolution")
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

            if not dropdown:
                return
            
            filepath = os.path.join(dirpath, dropdown + ".txt")
            print(filepath)
            f = open(filepath, 'w')
            f.write(str(viewnum) + "\n")
            f.write(widths + "\n")
            f.write(heights + "\n")
            f.write(weights + "\n")
            f.close()
            return

        alllist = [viewnum] + accordionlist +  [widths, heights, weights] + widthlist + heightlist + weightlist

        load_btn.click(
            loadpreset,
            [dropdown, widths, heights, weights],
            alllist,
        )
        
        save_btn.click(
            savepreset,
            [dropdown, viewnum, widths, heights, weights],
            [],
        )

        return [viewnum, widths, heights, weights]

    def run(self, p, viewnum: int, widths: str, heights: str, weights: str):
        
        widthlist = [int(s) for s in widths.split(',')] 
        heightlist = [int(s) for s in heights.split(',')]
        weightlist = [int(s) for s in weights.split(',')]
        
        addlist = weightlist
        # Init list for Random. [1,3,2,1] -> [1,4,6,7]
        for i in range(len(weightlist) - 1):
            addlist[i + 1] = addlist[i] + weightlist[i+1]


        processing.fix_seed(p)
        batch_count = p.n_iter

        p.batch_size = 1
        p.n_iter = 1

        info = None
        initial_seed = None
        initial_info = None
        initial_denoising_strength = p.denoising_strength

        grids = []
        all_images = []
        all_infos = []
        all_seeds = []
        original_prompt = p.prompt
        state.job_count =  batch_count

        history = []

        for n in range(batch_count):
            last_image = None


            # １画像出力.
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True

            random.seed(p.seed)
            randint = random.randint(1, int(addlist[viewnum-1]))
            for i in range(viewnum):
                if randint <= addlist[i]:
                    p.width = widthlist[i]
                    p.height = heightlist[i]
                    break

            state.job = f"Iteration 1/1, batch {n + 1}/{batch_count}"

            processed = processing.process_images(p)

            # Generation cancelled.
            if state.interrupted or state.stopping_generation:
                break

            all_seeds.append(processed.seed)
            all_infos.append(processed.info)
            p.seed = processed.seed + 1

            if state.skipped:
                break


            last_image = processed.images[0]

            if batch_count == 1:
                history.append(last_image)
                all_images.append(last_image)

            if batch_count > 1 and not state.skipped and not state.interrupted:
                history.append(last_image)
                all_images.append(last_image)

            if state.interrupted or state.stopping_generation:
                break

        if len(history) > 1:
            grid = images.image_grid(history, rows=1)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

            if opts.return_grid:
                grids.append(grid)

        all_images = grids + all_images

        processed = Processed(p, all_images, all_seeds=all_seeds, infotexts=all_infos)
        #processed = Processed(p, all_images, all_seeds, all_infos)

        return processed
