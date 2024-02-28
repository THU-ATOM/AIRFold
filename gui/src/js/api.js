var API = {
  api_prefix: 'http://' + window.location.hostname + ':8000/',
  get_requests: function () {
    return {
      method: "get",
      url: this.api_prefix + 'query'
    };
  },

  rerun_hashids: function (hash_ids) {
    return {
      method: "post",
      url: this.api_prefix + 'update/rerun',
      data: {
        hash_id: hash_ids
      }
    };
  },

  stop_hashid: function (hash_id) {
    return {
      method: "get",
      url: this.api_prefix + 'stop/' + hash_id,
    }
  },

  email_hashids: function (hash_ids) {
    return {
      method: "post",
      url: this.api_prefix + 'update/submit',
      data: {
        hash_id: hash_ids
      }
    };
  },

  update_lddt_hashids: function (hash_ids) {
    return {
      method: "post",
      url: this.api_prefix + 'update/lddt',
      data: {
        hash_id: hash_ids
      }
    };
  },

  update_plddt_hashids: function (hash_ids) {
    return {
      method: "post",
      url: this.api_prefix + 'update/gen_analysis',
      data: {
        hash_id: hash_ids
      }
    };
  },

  submit_request: function (request) {
    return {
      method: "post",
      url: this.api_prefix + 'insert/request',
      data: request
    };
  },

  update_comment: function (hash_id, comment) {
    return {
      method: "post",
      url: this.api_prefix + 'update/reserved/' + hash_id,
      data: { comment: comment }
    };
  },

  update_tags: function (hash_ids, tags, mode = "add") {
    // mode: add, remove, replace
    let tag_sep = ",";
    if (Array.isArray(tags)) {
      // concat tags with ','
      tags = tags.join(tag_sep);
    }
    console.log(tags)
    return {
      method: "post",
      url: this.api_prefix + 'update/tags',
      data: { tags: tags, hash_id: hash_ids, mode: mode }
    };
  },

  set_visibility: function (hash_id, visible) {
    return {
      method: "get",
      url: this.api_prefix + 'update/visible/' + hash_id + '?visible=' + (visible ? 1 : 0)
    }
  },

  image_url: function (path) {
    return this.api_prefix + 'file/png?file_path=' + path;
  },

  get_cameo_data: function (to_date) {
    return {
      method: "get",
      url: this.api_prefix + "cameo_data/" + to_date,
    }
  },

  get_casp_data: function () {
    return {
      method: "get",
      url: this.api_prefix + "casp_data",
    }
  },

  get_default_conf: function (mode) {
    return {
      method: "get",
      url: this.api_prefix + "/genconf/" + (mode ? mode : ""),
    }
  },

  cameo_url: function (cameo_obj) {
    return "https://www.cameo3d.org/modeling/targets/1-week/target/" + cameo_obj.date + "/" + cameo_obj.pdbid + "_" + cameo_obj.pdbid_chain + "/"
  },

  pdb_url: function (pdb_path) {
    return this.api_prefix + "file/text?file_path=" + pdb_path;
  },

  download_url: function (file_path) {
    return this.api_prefix + "file/download?file_path=" + file_path;
  },

  download_cameo: function (to_date) {
    return {
      method: "get",
      url: this.api_prefix + "update/cameo_gt/" + to_date,
    }
  },

  align_pdbs: function (pdbs) {
    return {
      method: "post",
      url: this.api_prefix + 'align',
      data: {
        pdbs: pdbs
      }
    };
  },

};

export { API };