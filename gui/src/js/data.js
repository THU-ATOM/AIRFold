import { API } from "./api";
import { split, forEach, map } from "lodash";
import axios from "axios";

var Data = {
  add_key: function (requests) {
    return map(requests, (request) => {
      request.key = request.hash_id;
      return request;
    });
  },

  // Parse model from rank_1_model_3_seed_0
  parse_model_name: function (model) {
    let model_split = split(model, "_");
    let model_name = "m" + model_split[model_split.length - 4];
    return model_name;
  },

  parse_data: function (data) {
    // deepcopy data
    var data_copy = JSON.parse(JSON.stringify(data));
    // iterate date_copy
    forEach(data_copy, (record) => {
      // parse date
      record.receive_time = this.parse_date(record.receive_time);
      // parse state_msg
      record.state_msg = this.parse_states(record.state_msg);
      // parse path_tree
      // record.path_tree = this.parse_pathtree(record.path_tree);
      // set last_update_time from last item's time of state_msg
      record.last_update_time =
        record.state_msg[record.state_msg.length - 1].time;
      // set release_date from request_json.time_stamp
      record.release_date = this.parse_date(record.request_json.time_stamp);
      record.analysis = this.parse_analysis(record);
      record.tags = this.parse_tags(record);
      // parse error
      if (record.error) {
        record.error = atob(record.error);
      }
    });
    return data_copy;
  },
  
  parse_states: function (states) {
    let state_objs = [];
    forEach(states, (state) => {
      let res = split(state, ":");
      state_objs.push({
        state: res[0],
        time: this.parse_date(res[1]),
      });
    });
    return state_objs;
  },

  parse_pathtree: function (path_tree) {
    // iterate key, value over path_tree
    let res = [];
    for (let key in path_tree) {
      let value = path_tree[key];
      if (typeof value === "object") {
        res.push({
          title: key,
          key: key,
          children: this.parse_pathtree(value),
        });
      } else {
        res.push({
          title: key + ": " + value,
          key: value,
        });
      }
    }
    return res;
  },

  parse_analysis: function (record) {
    if (record.path_tree) {
      var root = record.path_tree.alphafold.root;
      let conformations = [];
      forEach(record.path_tree.alphafold.model_files, (model_file) => {
        let name_splits = split(model_file.conformation, "/")
        let name = name_splits[name_splits.length - 1];
        // remove postfix from name
        name = name.substring(0, name.length - 4);
        conformations.push({
          url: API.image_url(model_file.conformation),
          name: name
        });
      });
      return {
        msa_coverage: API.image_url(root + "/msa_coverage.png"),
        predict_contacts: API.image_url(root + "/predicted_contacts.png"),
        predict_distogram: API.image_url(root + "/predicted_distogram.png"),
        predict_LDDT: API.image_url(root + "/predicted_LDDT.png"),
        conformations: conformations
      }
    } else {
      return null;
    }
  },

  // parse date from 20220416_173339 to Date
  parse_date: function (date) {
    return new Date(
      date.slice(0, 4),
      date.slice(4, 6) - 1,
      date.slice(6, 8),
      date.slice(9, 11),
      date.slice(11, 13),
      date.slice(13, 15)
    );
  },

  parse_config: function (record) {
    let config = JSON.parse(JSON.stringify(record.request_json));
    return config;
  },

  parse_tags: function (record) {
    let tags = [];
    if (record.request_json && record.reserved && record.reserved.tags) {
      tags = split(record.reserved.tags, ",");
    }
    return tags;
  },

  renew_config: function (config) {
    if (config.time_stamp) {
      delete config.time_stamp;
    }
    if (config.hash_id) {
      delete config.hash_id;
    }
    config.submit = false;
    config.name = split(config.name, "___")[0] + "___" + this.makeid(4);
    return config
  },

  makeid: function (length) {
    var result = "";
    var characters =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    var charactersLength = characters.length;
    for (var i = 0; i < length; i++) {
      result += characters.charAt(
        Math.floor(Math.random() * charactersLength)
      );
    }
    return result;
  },

  parse_pdbs_info: function (record, sort_by_name) {
    let res = [];
    if (record.path_tree && record.path_tree.alphafold) {
      forEach(record.path_tree.alphafold.model_files, (model_file, i) => {
        let item = {
          rank: i + 1,
          plddt: this.get_plddt(record, model_file.relaxed_pdb),
          plddt_img: this.get_img(model_file.image),
          lddt: this.get_lddt(record, model_file.relaxed_pdb),
          model_name: this.parse_model_name(model_file.relaxed_pdb),
          relaxed_pdb: model_file.relaxed_pdb,
          unrelaxed_pdb: model_file.unrelaxed_pdb,
        };
        res.push(item);
      });
      if (sort_by_name) {
        res.sort((a, b) => {
          return a.model_name.localeCompare(b.model_name);
        });
      }
    }
    return res;
  },

  parse_sequences: function (records) {
    let sequences = [];
    for (let i = 0; i < records.length; i++) {
      let record = records[i];
      let sequence = {
        sequence: record.request_json.sequence,
        label: record.name,
        id: i
      };
      sequences.push(sequence);
    }
    return sequences;
  },

  parse_msa: function (msa_path, on_load) {
    // let msa_path = record.path_tree.final_msa_fasta;
    axios.get(API.pdb_url(msa_path)).then((res) => {
      let sequences = [];
      let lines = res.data.split("\n");
      for (let i = 0; i <= lines.length / 2; i++) {
        let comment = lines[2 * i];
        let seq = lines[2 * i + 1];
        let label = split(comment, " ")[0].slice(1);
        if (label.length == 0) {
          label = "seq_" + i;
        }
        let sequence = {
          sequence: seq,
          label: label,
          id: i,
          comment: comment,
        };
        if (sequence.sequence) {
          sequences.push(sequence);
        }
      }
      on_load(sequences);
    });
  },

  parse_casp_data: function (data) {
    // iterate data
    let by_date = {};
    forEach(data, (record) => {
      let ddl = record["Human Exp."]
      if (by_date[ddl]) {
        by_date[ddl].push(record);
      }
      else {
        by_date[ddl] = [record];
      }
    });
    return by_date;
  },

  get_plddt: function (record, pdb_path) {
    let plddt = record.plddt;
    let res = 0;
    for (let key in plddt) {
      if (pdb_path.includes(key)) {
        res = plddt[key];
        return res;
      }
    }
    return 0;
  },

  get_lddt: function (record, pdb_path) {
    let lddt = record.lddt;
    let res = 0;
    for (let key in lddt) {
      if (pdb_path.includes(key)) {
        res = lddt[key];
        return res;
      }
    }
    return 0;
  },

  get_img: function (image_path) {
    return API.image_url(image_path);
  },

  // format Date from YYYYMMDD_HHMMSS
  format_date: function (date) {
    return (
      date.getFullYear() +
      "-" +
      (date.getMonth() + 1) +
      "-" +
      date.getDate() +
      " " +
      date.getHours() +
      ":" +
      date.getMinutes() +
      ":" +
      date.getSeconds()
    );
  },

};

export { Data };