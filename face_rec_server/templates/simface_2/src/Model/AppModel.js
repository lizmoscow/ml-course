var AppModel = Backbone.Model.extend({

    defaults: {
        uploadModel: new UploadModel(),
        profileList: new ProfileCollection()
    },

    initialize: function() {
        // this.uploadModel = new UploadModel();
        // this.profileList = new ProfileCollection();
    }
});